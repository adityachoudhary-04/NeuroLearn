import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from core.config import settings
from core.preprocessor import TextChunk

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    page_number: int
    doc_id: str
    metadata: Dict[str, Any]


class ChromaVectorStore:
    """ChromaDB-backed vector database with persistent storage, metadata filtering, and dimension-aware collections."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "pdf_rag_chunks"):
        self.persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.base_collection_name = collection_name
        self._collections: Dict[int, Any] = {}
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.is_available = True
        except Exception as e:
            print(f"Warning: ChromaDB initialization error: {e}. Falling back to in-memory store.")
            self.is_available = False
            self.memory_store: List[Dict[str, Any]] = []

    def _get_collection_for_dim(self, dim: int):
        if not self.is_available:
            return None
        if dim not in self._collections:
            name = f"{self.base_collection_name}_{dim}"
            self._collections[dim] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[dim]

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        if not chunks or not embeddings:
            return

        dim = len(embeddings[0])
        collection = self._get_collection_for_dim(dim)

        if self.is_available and collection is not None:
            ids = [c.chunk_id for c in chunks]
            documents = [c.text for c in chunks]
            metadatas = []
            for c in chunks:
                meta = {
                    "doc_id": c.doc_id,
                    "page_number": c.page_number,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "salience_score": float(c.salience_score),
                    "concepts_str": ", ".join(c.key_concepts[:5]),
                }
                # merge custom metadata
                for k, v in c.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                metadatas.append(meta)

            # Chroma batch upsert
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                collection.upsert(
                    ids=ids[i:i + batch_size],
                    embeddings=embeddings[i:i + batch_size],
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size]
                )
        else:
            # Memory store fallback
            for chunk, emb in zip(chunks, embeddings):
                self.memory_store.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "embedding": emb,
                    "metadata": chunk.metadata
                })

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        dim = len(query_embedding)
        collection = self._get_collection_for_dim(dim)

        if self.is_available and collection is not None:
            where_clause = {"doc_id": doc_id} if doc_id else None
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            search_results: List[SearchResult] = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)

                for c_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
                    similarity = 1.0 - float(dist) if dist is not None else 0.8
                    search_results.append(SearchResult(
                        chunk_id=c_id,
                        text=doc_text,
                        score=round(similarity, 4),
                        page_number=meta.get("page_number", 1),
                        doc_id=meta.get("doc_id", ""),
                        metadata=meta
                    ))
            return search_results
        else:
            # In-memory cosine similarity
            if not self.memory_store:
                return []
            
            filtered = [item for item in self.memory_store if (doc_id is None or item["doc_id"] == doc_id)]
            if not filtered:
                return []

            q_vec = np.array(query_embedding)
            q_norm = np.linalg.norm(q_vec)
            scored = []

            for item in filtered:
                emb_vec = np.array(item["embedding"])
                norm = np.linalg.norm(emb_vec)
                cos_sim = float(np.dot(q_vec, emb_vec) / (q_norm * norm + 1e-9))
                scored.append((cos_sim, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]

            return [
                SearchResult(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    score=round(score, 4),
                    page_number=item["page_number"],
                    doc_id=item["doc_id"],
                    metadata=item["metadata"]
                )
                for score, item in top
            ]

    def delete_document(self, doc_id: str):
        """Remove all chunks belonging to a document across all collections."""
        if self.is_available:
            for coll in self._collections.values():
                try:
                    coll.delete(where={"doc_id": doc_id})
                except Exception:
                    pass
            try:
                for coll in self.client.list_collections():
                    coll.delete(where={"doc_id": doc_id})
            except Exception:
                pass
        else:
            self.memory_store = [item for item in self.memory_store if item["doc_id"] != doc_id]

    def document_exists(self, doc_id: str) -> bool:
        """Return whether any persisted or in-memory chunk belongs to doc_id."""
        if self.is_available:
            try:
                for collection in self.client.list_collections():
                    result = collection.get(where={"doc_id": doc_id}, limit=1)
                    if result and result.get("ids"):
                        return True
            except Exception:
                return False
            return False
        return any(item["doc_id"] == doc_id for item in self.memory_store)

    def load_persisted_chunks(self) -> Dict[str, List[TextChunk]]:
        """Rebuild document chunk caches from Chroma after an app restart."""
        grouped: Dict[str, List[TextChunk]] = {}
        if not self.is_available:
            for item in self.memory_store:
                metadata = item.get("metadata", {})
                grouped.setdefault(item["doc_id"], []).append(TextChunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    text=item["text"],
                    page_number=item["page_number"],
                    start_char=int(metadata.get("start_char", 0)),
                    end_char=int(metadata.get("end_char", len(item["text"]))),
                    metadata=metadata,
                ))
            return grouped

        try:
            collections = self.client.list_collections()
        except Exception:
            return grouped

        seen_chunk_ids = set()
        for collection in collections:
            try:
                result = collection.get(include=["documents", "metadatas"])
            except Exception:
                continue
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            for chunk_id, text, metadata in zip(ids, documents, metadatas):
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                metadata = metadata or {}
                doc_id = str(metadata.get("doc_id", ""))
                if not doc_id:
                    continue
                concepts = metadata.get("key_concepts") or metadata.get("concepts_str", "")
                if isinstance(concepts, str):
                    concepts = [c.strip() for c in concepts.split(",") if c.strip()]
                grouped.setdefault(doc_id, []).append(TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text or "",
                    page_number=int(metadata.get("page_number", 1)),
                    start_char=int(metadata.get("start_char", 0)),
                    end_char=int(metadata.get("end_char", len(text or ""))),
                    key_concepts=concepts,
                    salience_score=float(metadata.get("salience_score", 0.0)),
                    metadata=metadata,
                ))

        for chunks in grouped.values():
            chunks.sort(key=lambda c: (c.page_number, c.start_char, c.chunk_id))
        return grouped

    def count(self) -> int:
        if self.is_available:
            total = 0
            try:
                for coll in self.client.list_collections():
                    total += coll.count()
            except Exception:
                for coll in self._collections.values():
                    total += coll.count()
            return total
        return len(self.memory_store)
