import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from core.config import settings
from core.rag_pipeline import pipeline_instance
from app.api.schemas import (
    URLIngestRequest,
    TextIngestRequest,
    QAGenerateRequest,
    MCQGenerateRequest,
    FlashcardGenerateRequest,
    SummaryGenerateRequest,
    ChatRequest,
    ConfigUpdateRequest,
    AnswerEvaluationRequest,
    TestPaperRequest,
)
from core.answer_evaluator import evaluate_answer
from core.test_paper import generate_test_paper, SUPPORTED_TYPES

api_router = APIRouter()

@api_router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "llm_provider": pipeline_instance.llm.provider_name,
        "llm_model": pipeline_instance.llm.model_name,
        "embedding_provider": type(pipeline_instance.embedding).__name__,
        "documents_indexed": len(pipeline_instance.doc_registry),
        "total_chunks": pipeline_instance.vector_store.count()
    }

@api_router.get("/config")
def get_config():
    return {
        "llm_provider": pipeline_instance.llm.provider_name,
        "llm_model": pipeline_instance.llm.model_name,
        "embedding_provider": pipeline_instance.embedding_provider_name,
        "embedding_class": type(pipeline_instance.embedding).__name__,
        "embedding_model": getattr(pipeline_instance.embedding, "model_name", None),
        "embedding_backend": getattr(pipeline_instance.embedding, "backend_name", type(pipeline_instance.embedding).__name__),
        "salience_method": pipeline_instance.salience_method,
        "has_gemini_key": bool(settings.GEMINI_API_KEY),
        "has_groq_key": bool(settings.GROQ_API_KEY),
        "has_openai_key": bool(settings.OPENAI_API_KEY),
        "has_anthropic_key": bool(settings.ANTHROPIC_API_KEY),
    }

@api_router.post("/config")
def update_config(req: ConfigUpdateRequest):
    pipeline_instance.update_providers(
        llm_name=req.llm_provider,
        llm_key=req.llm_api_key,
        llm_model=req.llm_model,
        emb_name=req.embedding_provider,
        emb_key=req.embedding_api_key,
        emb_model=req.embedding_model,
        salience_method=req.salience_method
    )
    return {
        "message": "Configuration successfully updated",
        "active_llm": pipeline_instance.llm.provider_name,
        "active_model": pipeline_instance.llm.model_name,
        "active_embedding": type(pipeline_instance.embedding).__name__,
        "active_salience_method": pipeline_instance.salience_method
    }

@api_router.post("/documents/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    salience_method: str = Form("tfidf"),
    embedding_provider: str = Form("bert")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for file upload.")

    dest_path = settings.UPLOAD_DIR / file.filename
    try:
        if embedding_provider:
            pipeline_instance.update_providers(emb_name=embedding_provider, salience_method=salience_method)

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        meta = pipeline_instance.ingest_pdf(dest_path, salience_method=salience_method)
        return {
            "message": "PDF uploaded and indexed successfully",
            "document": meta
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@api_router.post("/documents/url")
async def ingest_url(req: URLIngestRequest):
    try:
        if req.embedding_provider:
            pipeline_instance.update_providers(emb_name=req.embedding_provider, salience_method=req.salience_method)
        meta = await pipeline_instance.ingest_url(req.url, salience_method=req.salience_method)
        return {
            "message": "Webpage scraped and indexed successfully",
            "document": meta
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")

@api_router.post("/documents/text")
def ingest_text(req: TextIngestRequest):
    try:
        if req.embedding_provider:
            pipeline_instance.update_providers(emb_name=req.embedding_provider, salience_method=req.salience_method)
        meta = pipeline_instance.ingest_raw_text(req.text, title=req.title, salience_method=req.salience_method)
        return {
            "message": "Raw text indexed successfully",
            "document": meta
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

@api_router.get("/documents")
def list_documents():
    return {
        "documents": pipeline_instance.list_documents()
    }

@api_router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    success = pipeline_instance.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Document {doc_id} successfully deleted"}

@api_router.post("/generate/qa")
def generate_qa(req: QAGenerateRequest):
    chunks = pipeline_instance.get_document_chunks(req.doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found or has no chunks.")

    qa_pairs = pipeline_instance.generate_qa(
        doc_id=req.doc_id,
        num_questions=req.num_questions,
        target_difficulty=req.target_difficulty,
        target_bloom=req.target_bloom,
        topic=req.topic
    )
    return {
        "doc_id": req.doc_id,
        "count": len(qa_pairs),
        "qa_pairs": qa_pairs
    }

@api_router.post("/generate/mcq")
def generate_mcqs(req: MCQGenerateRequest):
    chunks = pipeline_instance.get_document_chunks(req.doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found.")

    mcqs = pipeline_instance.generate_mcqs(
        doc_id=req.doc_id,
        num_questions=req.num_questions,
        target_difficulty=req.target_difficulty,
        topic=req.topic
    )
    return {
        "doc_id": req.doc_id,
        "count": len(mcqs),
        "mcqs": mcqs
    }

@api_router.post("/generate/flashcards")
def generate_flashcards(req: FlashcardGenerateRequest):
    chunks = pipeline_instance.get_document_chunks(req.doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found.")

    flashcards = pipeline_instance.generate_flashcards(
        doc_id=req.doc_id,
        num_cards=req.num_cards
    )
    return {
        "doc_id": req.doc_id,
        "count": len(flashcards),
        "flashcards": flashcards
    }

@api_router.post("/generate/summary")
def generate_summary(req: SummaryGenerateRequest):
    chunks = pipeline_instance.get_document_chunks(req.doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found.")

    summary = pipeline_instance.generate_summary(doc_id=req.doc_id)
    return {
        "doc_id": req.doc_id,
        "summary": summary
    }

@api_router.post("/chat")
def chat(req: ChatRequest):
    try:
        history = [msg.model_dump() for msg in req.chat_history] if req.chat_history else None
        response = pipeline_instance.chat(
            query=req.query,
            doc_id=req.doc_id,
            chat_history=history,
            top_k=req.top_k
        )
        return {
            "answer": response.answer,
            "citations": response.citations,
            "confidence_level": response.confidence_level
        }
    except Exception as exc:
        # Always return JSON so the browser can show a useful error instead of
        # failing again while trying to parse an HTML/plain-text 500 response.
        raise HTTPException(
            status_code=502,
            detail=f"Chat service unavailable: {str(exc)[:240]}"
        ) from exc

@api_router.post("/evaluate/answer")
def evaluate_descriptive_answer(req: AnswerEvaluationRequest):
    chunks = pipeline_instance.get_document_chunks(req.doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found or has no indexed content.")
    source = "\n\n".join(
        f"[Page {chunk.page_number}] {chunk.text}" for chunk in chunks
        if req.page_number is None or chunk.page_number == req.page_number
    )
    return evaluate_answer(
        pipeline_instance.llm,
        question=req.question,
        reference_answer=req.reference_answer,
        student_answer=req.student_answer,
        source_context=source[:24000],
        max_marks=req.max_marks,
        rubric=req.rubric or "Assess conceptual correctness, factual accuracy, completeness, and reasoning.",
    )

@api_router.post("/generate/test-paper")
def generate_custom_test_paper(req: TestPaperRequest):
    chunks = pipeline_instance.get_document_chunks(req.doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found or has no indexed content.")
    invalid = set(req.question_counts) - SUPPORTED_TYPES
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unsupported question type(s): {', '.join(sorted(invalid))}")
    counts = {key: int(value) for key, value in req.question_counts.items() if int(value) > 0}
    total = sum(counts.values())
    if total < 1 or total > 50:
        raise HTTPException(status_code=422, detail="Total question count must be between 1 and 50.")
    return generate_test_paper(
        pipeline_instance.llm,
        chunks=chunks,
        title=req.title,
        subject=req.subject,
        question_counts=counts,
        difficulty=req.difficulty,
        duration_minutes=req.duration_minutes,
        marks=req.marks,
        include_answer_key=req.include_answer_key,
        include_explanations=req.include_explanations,
        include_rubrics=req.include_rubrics,
        shuffle=req.shuffle,
        topic=req.topic,
    )
