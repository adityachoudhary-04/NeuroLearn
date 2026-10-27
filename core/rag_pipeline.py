import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from core.config import settings
from core.extractor import DocumentExtractor, DocumentExtract
from core.preprocessor import TextPreprocessor, TextChunk
from core.embedding_provider import BaseEmbeddingProvider, BertEmbeddingProvider, get_embedding_provider
from core.llm_provider import BaseLLMProvider, get_llm_provider
from core.vector_store import ChromaVectorStore, SearchResult
from core.generators.qa_generator import QAGenerator, QAPair
from core.generators.mcq_generator import MCQGenerator, MCQ
from core.generators.flashcard_generator import FlashcardGenerator, Flashcard
from core.generators.summary_generator import SummaryGenerator, DocumentSummary
from core.generators.chat_engine import ChatEngine, ChatResponse

@dataclass
class DocumentMetadata:
    doc_id: str
    title: str
    source_type: str
    total_pages: int
    total_chunks: int
    created_at: float
    file_path: Optional[str] = None
    url: Optional[str] = None
    summary_preview: Optional[str] = None


class RAGPipeline:
    """Master orchestrator for document ingestion, indexing, retrieval, and generation."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        salience_method: Optional[str] = None
    ):
        self.embedding = embedding_provider or get_embedding_provider()
        self.embedding_provider_name = self._effective_embedding_name(self.embedding)
        self.llm = llm_provider or get_llm_provider()
        self.vector_store = vector_store or ChromaVectorStore()
        self.salience_method = salience_method or settings.DEFAULT_SALIENCE_METHOD

        # In-memory document registry (cached chunk lookup and metadata)
        self.doc_registry: Dict[str, DocumentMetadata] = {}
        self.doc_chunks: Dict[str, List[TextChunk]] = {}

        # Chroma persists vectors, so restore the in-memory document catalogue
        # from those records as well.  Without this, every restart made the UI
        # report that no documents existed.
        self._restore_persisted_documents()

        # Initialize sub-generators
        self._refresh_generators()

    def update_providers(
        self,
        llm_name: Optional[str] = None,
        llm_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        emb_name: Optional[str] = None,
        emb_key: Optional[str] = None,
        emb_model: Optional[str] = None,
        salience_method: Optional[str] = None
    ):
        """Allows dynamic switching of LLM, Embedding, or Salience providers at runtime."""
        if llm_name or llm_key or llm_model:
            self.llm = get_llm_provider(name=llm_name, api_key=llm_key, model=llm_model)
        if emb_name or emb_key or emb_model:
            self.embedding = get_embedding_provider(name=emb_name, api_key=emb_key, model=emb_model)
            self.embedding_provider_name = self._effective_embedding_name(self.embedding)
        if salience_method:
            self.salience_method = salience_method.lower().strip()
        self._refresh_generators()

    @staticmethod
    def _effective_embedding_name(provider: BaseEmbeddingProvider) -> str:
        """Report the provider actually instantiated, not the requested option."""
        if isinstance(provider, BertEmbeddingProvider):
            return "bert"
        if provider.__class__.__name__ == "LocalHashEmbeddingProvider":
            return "local"
        if provider.__class__.__name__ == "OpenAIEmbeddingProvider":
            return "openai"
        if provider.__class__.__name__ == "GeminiEmbeddingProvider":
            return "gemini"
        return provider.__class__.__name__.replace("EmbeddingProvider", "").lower()

    def _refresh_generators(self):
        self.qa_gen = QAGenerator(self.llm)
        self.mcq_gen = MCQGenerator(self.llm)
        self.flashcard_gen = FlashcardGenerator(self.llm)
        self.summary_gen = SummaryGenerator(self.llm)
        self.chat_engine = ChatEngine(self.llm)

    def _restore_persisted_documents(self):
        persisted = self.vector_store.load_persisted_chunks()
        for doc_id, chunks in persisted.items():
            if not chunks:
                continue
            first = chunks[0]
            title = str(first.metadata.get("title") or doc_id)
            source_type = str(first.metadata.get("source_type") or "unknown")
            self.doc_chunks[doc_id] = chunks
            self.doc_registry[doc_id] = DocumentMetadata(
                doc_id=doc_id,
                title=title,
                source_type=source_type,
                total_pages=max(c.page_number for c in chunks),
                total_chunks=len(chunks),
                created_at=0.0,
                summary_preview=first.text[:200],
            )

    def _chunks_from_results(self, doc_id: str, results: List[SearchResult]) -> List[TextChunk]:
        """Resolve search results back to TextChunk objects for generators."""
        cached = {chunk.chunk_id: chunk for chunk in self.get_document_chunks(doc_id)}
        resolved: List[TextChunk] = []
        for result in results:
            chunk = cached.get(result.chunk_id)
            if chunk is None:
                chunk = TextChunk(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    text=result.text,
                    page_number=result.page_number,
                    start_char=int(result.metadata.get("start_char", 0)),
                    end_char=int(result.metadata.get("end_char", len(result.text))),
                    salience_score=float(result.metadata.get("salience_score", 0.0)),
                    metadata=result.metadata,
                )
            resolved.append(chunk)
        return resolved

    def ingest_pdf(self, file_path: Path, salience_method: Optional[str] = None) -> DocumentMetadata:
        """Extracts, chunks, embeds, and indexes a PDF document."""
        extract = DocumentExtractor.extract_from_pdf(file_path)
        return self._process_and_index(extract, file_path=str(file_path), salience_method=salience_method)

    async def ingest_url(self, url: str, salience_method: Optional[str] = None) -> DocumentMetadata:
        """Fetches, cleans, chunks, embeds, and indexes a webpage."""
        extract = await DocumentExtractor.extract_from_url(url)
        return self._process_and_index(extract, url=url, salience_method=salience_method)

    def ingest_raw_text(self, text: str, title: str = "Pasted Text", salience_method: Optional[str] = None) -> DocumentMetadata:
        """Chunks, embeds, and indexes raw pasted text."""
        extract = DocumentExtractor.extract_from_raw_text(text, title=title)
        return self._process_and_index(extract, salience_method=salience_method)

    def _process_and_index(
        self,
        extract: DocumentExtract,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        salience_method: Optional[str] = None
    ) -> DocumentMetadata:
        # 1. Chunk document and extract concepts using selected method
        active_salience_method = salience_method or self.salience_method
        chunks = TextPreprocessor.chunk_document(
            extract,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            salience_method=active_salience_method,
            embedding_provider=self.embedding
        )

        if not chunks:
            # Handle empty document case
            dummy_chunk = TextChunk(
                chunk_id=f"{extract.document_id}_p1_c1",
                doc_id=extract.document_id,
                text="Document has no readable text.",
                page_number=1,
                start_char=0,
                end_char=0
            )
            chunks = [dummy_chunk]

        # 2. Generate embeddings
        texts = [c.text for c in chunks]
        try:
            embeddings = self.embedding.embed_batch(texts)
        except Exception as embedding_error:
            # A provider can initialize successfully but still fail on the
            # first API request (invalid model, quota, or transient outage).
            # Keep ingestion usable by switching this indexing operation to
            # the local BERT/hash implementation instead of returning a 500.
            if isinstance(self.embedding, BertEmbeddingProvider):
                raise embedding_error
            print(f"Warning: embedding request failed ({embedding_error}); using local BERT fallback.")
            self.embedding = BertEmbeddingProvider(model_name="all-MiniLM-L6-v2")
            self.embedding_provider_name = self._effective_embedding_name(self.embedding)
            embeddings = self.embedding.embed_batch(texts)

        # 3. Store in vector database
        self.vector_store.add_chunks(chunks, embeddings)

        # 4. Save in registry
        meta = DocumentMetadata(
            doc_id=extract.document_id,
            title=extract.title,
            source_type=extract.source_type,
            total_pages=extract.total_pages,
            total_chunks=len(chunks),
            created_at=time.time(),
            file_path=file_path,
            url=url,
            summary_preview=chunks[0].text[:200] if chunks else ""
        )
        self.doc_registry[extract.document_id] = meta
        self.doc_chunks[extract.document_id] = chunks

        return meta

    def get_document_chunks(self, doc_id: str) -> List[TextChunk]:
        return self.doc_chunks.get(doc_id, [])

    def list_documents(self) -> List[DocumentMetadata]:
        return list(self.doc_registry.values())

    def delete_document(self, doc_id: str) -> bool:
        existed = (
            doc_id in self.doc_registry
            or doc_id in self.doc_chunks
            or self.vector_store.document_exists(doc_id)
        )
        if not existed:
            return False
        if doc_id in self.doc_registry:
            del self.doc_registry[doc_id]
        if doc_id in self.doc_chunks:
            del self.doc_chunks[doc_id]
        self.vector_store.delete_document(doc_id)
        return True

    def generate_qa(
        self,
        doc_id: str,
        num_questions: int = 5,
        target_difficulty: Optional[str] = None,
        target_bloom: Optional[str] = None,
        topic: Optional[str] = None
    ) -> List[QAPair]:
        if topic:
            query_emb = self.embedding.embed_text(topic)
            results = self.vector_store.query(
                query_embedding=query_emb,
                top_k=min(10, num_questions * 2),
                doc_id=doc_id
            )
            chunks = self._chunks_from_results(doc_id, results) if results else self.get_document_chunks(doc_id)
        else:
            chunks = self.get_document_chunks(doc_id)

        return self.qa_gen.generate_from_chunks(
            chunks=chunks,
            num_questions=num_questions,
            target_difficulty=target_difficulty,
            target_bloom=target_bloom
        )

    def generate_mcqs(
        self,
        doc_id: str,
        num_questions: int = 5,
        target_difficulty: Optional[str] = None,
        topic: Optional[str] = None
    ) -> List[MCQ]:
        if topic:
            query_emb = self.embedding.embed_text(topic)
            results = self.vector_store.query(
                query_embedding=query_emb,
                top_k=min(10, num_questions * 2),
                doc_id=doc_id
            )
            chunks = self._chunks_from_results(doc_id, results) if results else self.get_document_chunks(doc_id)
        else:
            chunks = self.get_document_chunks(doc_id)

        return self.mcq_gen.generate_from_chunks(
            chunks=chunks,
            num_questions=num_questions,
            target_difficulty=target_difficulty
        )

    def generate_flashcards(
        self,
        doc_id: str,
        num_cards: int = 6
    ) -> List[Flashcard]:
        chunks = self.get_document_chunks(doc_id)
        return self.flashcard_gen.generate_from_chunks(chunks=chunks, num_cards=num_cards)

    def generate_summary(self, doc_id: str) -> DocumentSummary:
        chunks = self.get_document_chunks(doc_id)
        meta = self.doc_registry.get(doc_id)
        title = meta.title if meta else "Document"
        return self.summary_gen.generate_from_chunks(chunks=chunks, doc_title=title)

    def chat(
        self,
        query: str,
        doc_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5
    ) -> ChatResponse:
        try:
            # Embed user query
            query_emb = self.embedding.embed_text(query)

            # Retrieve relevant chunks from vector store
            results = self.vector_store.query(
                query_embedding=query_emb,
                top_k=top_k,
                doc_id=doc_id
            )
        except Exception as exc:
            # Retrieval failures should be rendered as an assistant response,
            # not an unparseable server 500 that breaks the chat UI.
            return ChatResponse(
                answer=(
                    "I couldn't search this document right now. "
                    "Please retry, or re-index the document if the problem continues."
                ),
                citations=[],
                confidence_level="Low"
            )

        # Pass retrieved context to chat engine
        return self.chat_engine.answer_query(
            query=query,
            retrieved_chunks=results,
            chat_history=chat_history
        )


# Global singleton instance for the FastAPI app
pipeline_instance = RAGPipeline()
