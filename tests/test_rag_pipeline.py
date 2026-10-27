import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.extractor import DocumentExtractor
from core.preprocessor import TextPreprocessor
from core.embedding_provider import get_embedding_provider
from core.vector_store import ChromaVectorStore
from core.rag_pipeline import RAGPipeline
from core.generators.qa_generator import QAGenerator
from core.generators.mcq_generator import MCQGenerator
from core.generators.flashcard_generator import FlashcardGenerator
from core.generators.summary_generator import SummaryGenerator
from core.generators.chat_engine import ChatEngine
from core.llm_provider import MockLLMProvider

def test_full_pipeline():
    print("=" * 60)
    print("Testing NeuroLearn PDF RAG & Q/A Pipeline...")
    print("=" * 60)

    # 1. Sample document text simulating extracted PDF pages
    sample_text = """
Retrieval-Augmented Generation (RAG) is an architectural framework that combines information retrieval 
with large language models (LLMs). By fetching relevant passages from external knowledge bases—such as 
academic PDFs, research papers, and technical documentation—RAG mitigates hallucinations and keeps 
answers strictly grounded in factual context.

Bloom's Taxonomy is a hierarchical model used to classify educational learning objectives into levels of 
complexity and specificity: Remember, Understand, Apply, Analyze, Evaluate, and Create. In question-generation 
pipelines, applying Bloom's Taxonomy ensures a diverse distribution of questions ranging from simple recall 
to complex analytical inquiry.

Vector databases, such as ChromaDB and FAISS, index high-dimensional embeddings computed by dense transformer 
models. When a user issues a query, cosine similarity search retrieves the top-k most semantically relevant 
chunks to feed into the prompt context window.
"""

    print("\n[Step 1] Ingesting text extract...")
    extract = DocumentExtractor.extract_from_raw_text(sample_text, title="RAG & Blooms Architecture")
    assert extract.total_pages == 1
    assert len(extract.full_text) > 100
    print(f" -> Extracted title: {extract.title}, chars: {len(extract.full_text)}")

    print("\n[Step 2] Chunking & TF-IDF concept salience...")
    chunks = TextPreprocessor.chunk_document(extract, chunk_size=400, chunk_overlap=50)
    print(f" -> Generated {len(chunks)} chunks.")
    for idx, c in enumerate(chunks):
        print(f"    Chunk {idx+1}: concepts={c.key_concepts}, salience={c.salience_score:.3f}")
    assert len(chunks) >= 1

    print("\n[Step 3] Testing Embedding Provider...")
    emb_provider = get_embedding_provider("local")
    embeddings = emb_provider.embed_batch([c.text for c in chunks])
    assert len(embeddings) == len(chunks)
    assert len(embeddings[0]) == emb_provider.dimension
    print(f" -> Embeddings generated with dimension: {len(embeddings[0])}")

    print("\n[Step 4] Testing Vector Store Indexing & Retrieval...")
    test_db_dir = ROOT / "data" / "test_chroma"
    test_db_dir.mkdir(parents=True, exist_ok=True)
    vstore = ChromaVectorStore(persist_dir=str(test_db_dir), collection_name="test_collection")
    vstore.add_chunks(chunks, embeddings)

    q_vec = emb_provider.embed_text("How does Bloom's Taxonomy classify learning objectives?")
    search_results = vstore.query(q_vec, top_k=2)
    print(f" -> Retrieved {len(search_results)} chunks for test query.")
    for res in search_results:
        print(f"    Score {res.score:.3f} | Page {res.page_number} | Text: {res.text[:70]}...")
    assert len(search_results) > 0

    print("\n[Step 5] Testing Master RAG Pipeline...")
    pipeline = RAGPipeline(embedding_provider=emb_provider, vector_store=vstore)
    doc_meta = pipeline.ingest_raw_text(sample_text, title="Test Pipeline Doc")
    print(f" -> Indexed doc_id: {doc_meta.doc_id}, total chunks: {doc_meta.total_chunks}")

    print("\n[Step 6] Testing Q&A Generation with Bloom's & Difficulty...")
    qa_list = pipeline.generate_qa(doc_meta.doc_id, num_questions=2)
    print(f" -> Generated {len(qa_list)} Q&A pairs:")
    for qa in qa_list:
        print(f"    [{qa.blooms_level} | {qa.difficulty}] Q: {qa.question}")
        print(f"    A: {qa.answer[:80]}...")

    print("\n[Step 7] Testing MCQ Quiz Generation...")
    mcqs = pipeline.generate_mcqs(doc_meta.doc_id, num_questions=2)
    print(f" -> Generated {len(mcqs)} MCQs:")
    for mcq in mcqs:
        print(f"    Q: {mcq.question}")
        print(f"    Options: {mcq.options}")
        print(f"    Correct: {mcq.correct_answer} (Expl: {mcq.explanation[:60]}...)")

    print("\n[Step 8] Testing Flashcards Generation...")
    cards = pipeline.generate_flashcards(doc_meta.doc_id, num_cards=2)
    print(f" -> Generated {len(cards)} Flashcards:")
    for card in cards:
        print(f"    Front: {card.front} | Back: {card.back[:50]}...")

    print("\n[Step 9] Testing Summary Generation...")
    summary = pipeline.generate_summary(doc_meta.doc_id)
    print(f" -> Executive Summary: {summary.executive_summary[:100]}...")
    print(f" -> Takeaways: {len(summary.key_takeaways)}")

    print("\n[Step 10] Testing RAG Chat with Citations...")
    chat_resp = pipeline.chat("Explain the benefits of RAG in preventing hallucinations.", doc_id=doc_meta.doc_id)
    print(f" -> Chat Answer: {chat_resp.answer[:120]}...")
    print(f" -> Citations: {len(chat_resp.citations)} (Confidence: {chat_resp.confidence_level})")

    print("\n" + "=" * 60)
    print("ALL 10 PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


def test_restart_persistence_and_topic_generation(tmp_path):
    """Regression coverage for the runtime paths used by the dashboard."""
    db_dir = tmp_path / "persistent_chroma"
    embedding = get_embedding_provider("local")
    store = ChromaVectorStore(persist_dir=str(db_dir), collection_name="runtime_test")
    pipeline = RAGPipeline(
        llm_provider=MockLLMProvider(),
        embedding_provider=embedding,
        vector_store=store,
    )

    meta = pipeline.ingest_raw_text(
        "Photosynthesis converts light energy into chemical energy. "
        "Chlorophyll absorbs light and supports glucose production.",
        title="Photosynthesis Notes",
    )

    # Topic filtering previously tried to access SearchResult.chunk, a field
    # that does not exist.
    assert pipeline.generate_qa(meta.doc_id, num_questions=1, topic="chlorophyll")
    assert pipeline.generate_mcqs(meta.doc_id, num_questions=1, topic="light energy")

    # A new pipeline process should rebuild its document list from Chroma.
    restored = RAGPipeline(
        llm_provider=MockLLMProvider(),
        embedding_provider=embedding,
        vector_store=ChromaVectorStore(
            persist_dir=str(db_dir), collection_name="runtime_test"
        ),
    )
    assert meta.doc_id in {doc.doc_id for doc in restored.list_documents()}
    assert restored.get_document_chunks(meta.doc_id)
    assert restored.delete_document("missing-document") is False

if __name__ == "__main__":
    test_full_pipeline()
