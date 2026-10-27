import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.extractor import DocumentExtractor
from core.preprocessor import TextPreprocessor
from core.embedding_provider import get_embedding_provider
from core.rag_pipeline import RAGPipeline

def test_all_salience_methods():
    print("=" * 65)
    print("Testing Content Selection & Salience Extraction Methods")
    print("=" * 65)

    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an architectural framework that combines "
        "information retrieval with large language models. Dense vector embeddings and cosine "
        "similarity search enable accurate semantic matching. Bloom's Taxonomy classifies cognitive "
        "learning objectives into distinct hierarchical tiers from simple recall to critical synthesis."
    )

    extract = DocumentExtractor.extract_from_raw_text(sample_text, title="Salience Comparison Test")
    emb_provider = get_embedding_provider("local")

    # 1. Test TF-IDF
    print("\n[Method 1: Statistical TF-IDF]")
    chunks_tfidf = TextPreprocessor.chunk_document(
        extract, chunk_size=300, chunk_overlap=50, salience_method="tfidf"
    )
    for i, c in enumerate(chunks_tfidf):
        print(f"  Chunk {i+1}: concepts={c.key_concepts}, salience={c.salience_score:.4f}, method={c.metadata.get('salience_method')}")
    assert len(chunks_tfidf) > 0
    assert len(chunks_tfidf[0].key_concepts) > 0
    assert chunks_tfidf[0].metadata.get("salience_method") == "tfidf"

    # 2. Test TextRank (Graph-based PageRank)
    print("\n[Method 2: Graph-Based TextRank (PageRank Centrality)]")
    chunks_tr = TextPreprocessor.chunk_document(
        extract, chunk_size=300, chunk_overlap=50, salience_method="textrank"
    )
    for i, c in enumerate(chunks_tr):
        print(f"  Chunk {i+1}: concepts={c.key_concepts}, salience={c.salience_score:.4f}, method={c.metadata.get('salience_method')}")
    assert len(chunks_tr) > 0
    assert len(chunks_tr[0].key_concepts) > 0
    assert chunks_tr[0].metadata.get("salience_method") == "textrank"

    # 3. Test Embedding-Based (Semantic Cosine Similarity)
    print("\n[Method 3: Dense Vector Semantic Similarity (KeyBERT-style)]")
    chunks_emb = TextPreprocessor.chunk_document(
        extract, chunk_size=300, chunk_overlap=50, salience_method="embedding", embedding_provider=emb_provider
    )
    for i, c in enumerate(chunks_emb):
        print(f"  Chunk {i+1}: concepts={c.key_concepts}, salience={c.salience_score:.4f}, method={c.metadata.get('salience_method')}")
    assert len(chunks_emb) > 0
    assert len(chunks_emb[0].key_concepts) > 0
    assert chunks_emb[0].metadata.get("salience_method") == "embedding"

    # 4. Test RAGPipeline Integration with selection
    print("\n[Testing RAGPipeline Ingest with Salience Method Selection]")
    pipeline = RAGPipeline(embedding_provider=emb_provider)
    
    doc_tr = pipeline.ingest_raw_text(sample_text, title="Doc with TextRank", salience_method="textrank")
    print(f"  -> Ingested doc with TextRank: {doc_tr.doc_id}, chunks: {doc_tr.total_chunks}")
    
    doc_emb = pipeline.ingest_raw_text(sample_text, title="Doc with Embedding", salience_method="embedding")
    print(f"  -> Ingested doc with Embedding: {doc_emb.doc_id}, chunks: {doc_emb.total_chunks}")

    print("\n" + "=" * 65)
    print("ALL 3 METHODS TESTED & FUNCTIONAL!")
    print("=" * 65)

if __name__ == "__main__":
    test_all_salience_methods()
