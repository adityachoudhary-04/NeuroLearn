from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class URLIngestRequest(BaseModel):
    url: str
    salience_method: Optional[str] = None  # "tfidf", "textrank", "embedding"
    embedding_provider: Optional[str] = None

class TextIngestRequest(BaseModel):
    title: str = "Pasted Notes"
    text: str
    salience_method: Optional[str] = None
    embedding_provider: Optional[str] = None

class QAGenerateRequest(BaseModel):
    doc_id: str
    num_questions: int = Field(default=10, ge=1, le=50)
    target_difficulty: Optional[str] = None  # "Easy", "Medium", "Hard"
    target_bloom: Optional[str] = None  # "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
    topic: Optional[str] = None

class MCQGenerateRequest(BaseModel):
    doc_id: str
    num_questions: int = Field(default=10, ge=1, le=50)
    target_difficulty: Optional[str] = None
    topic: Optional[str] = None

class FlashcardGenerateRequest(BaseModel):
    doc_id: str
    num_cards: int = Field(default=6, ge=1, le=25)

class SummaryGenerateRequest(BaseModel):
    doc_id: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = None
    top_k: int = Field(default=5, ge=1, le=10)

class AnswerEvaluationRequest(BaseModel):
    doc_id: str
    question: str
    reference_answer: str
    student_answer: str
    page_number: Optional[int] = None
    max_marks: float = Field(default=10, ge=0.5, le=100)
    rubric: Optional[str] = None

class TestPaperRequest(BaseModel):
    doc_id: str
    title: str = "Custom Test Paper"
    subject: str = ""
    question_counts: Dict[str, int]
    difficulty: str = "Mixed"
    duration_minutes: int = Field(default=60, ge=5, le=600)
    marks: Dict[str, float] = Field(default_factory=dict)
    topic: str = "All topics"
    include_answer_key: bool = True
    include_explanations: bool = True
    include_rubrics: bool = True
    shuffle: bool = False

class ConfigUpdateRequest(BaseModel):
    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model: Optional[str] = None
    salience_method: Optional[str] = None  # "tfidf", "textrank", "embedding"
