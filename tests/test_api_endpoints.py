import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import app

def test_api():
    print("Testing FastAPI endpoints with TestClient...")
    client = TestClient(app)

    # 1. Health check
    res = client.get("/api/health")
    print("GET /api/health ->", res.status_code, res.json())
    assert res.status_code == 200

    # 2. Config check
    res = client.get("/api/config")
    print("GET /api/config ->", res.status_code, res.json())
    assert res.status_code == 200

    # 3. Text Ingest
    res = client.post("/api/documents/text", json={
        "title": "Quantum Computing Fundamentals",
        "text": "Quantum computers use qubits instead of classical bits. Qubits can exist in a superposition of states, allowing quantum algorithms like Shor's algorithm and Grover's algorithm to solve certain problems exponentially faster than classical computers."
    })
    print("POST /api/documents/text ->", res.status_code, res.json())
    assert res.status_code == 200
    doc_id = res.json()["document"]["doc_id"]

    # 4. List Documents
    res = client.get("/api/documents")
    print("GET /api/documents ->", res.status_code, len(res.json()["documents"]))
    assert res.status_code == 200

    # 5. Generate Q&A
    res = client.post("/api/generate/qa", json={"doc_id": doc_id, "num_questions": 2})
    print("POST /api/generate/qa ->", res.status_code, len(res.json()["qa_pairs"]))
    assert res.status_code == 200

    # 6. Generate MCQs
    res = client.post("/api/generate/mcq", json={"doc_id": doc_id, "num_questions": 2})
    print("POST /api/generate/mcq ->", res.status_code, len(res.json()["mcqs"]))
    assert res.status_code == 200

    # 7. Generate Flashcards
    res = client.post("/api/generate/flashcards", json={"doc_id": doc_id, "num_cards": 2})
    print("POST /api/generate/flashcards ->", res.status_code, len(res.json()["flashcards"]))
    assert res.status_code == 200

    # 8. Generate Summary
    res = client.post("/api/generate/summary", json={"doc_id": doc_id})
    print("POST /api/generate/summary ->", res.status_code, bool(res.json()["summary"]))
    assert res.status_code == 200

    # 9. RAG Chat
    res = client.post("/api/chat", json={"query": "What do quantum computers use?", "doc_id": doc_id})
    print("POST /api/chat ->", res.status_code, res.json()["confidence_level"])
    assert res.status_code == 200

    print("\nALL API ENDPOINTS TESTED AND FUNCTIONAL!")

if __name__ == "__main__":
    test_api()
