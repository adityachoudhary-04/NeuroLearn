import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.llm_provider import BaseLLMProvider
from core.vector_store import SearchResult

@dataclass
class Citation:
    page_number: int
    chunk_id: str
    relevance_score: float
    excerpt: str

@dataclass
class ChatResponse:
    answer: str
    citations: List[Citation]
    confidence_level: str  # "High", "Medium", "Low"


class ChatEngine:
    """Conversational RAG assistant grounded strictly in document retrieval."""

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def answer_query(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        if not retrieved_chunks:
            return ChatResponse(
                answer="I could not find relevant information in the uploaded document to answer this query.",
                citations=[],
                confidence_level="Low"
            )

        # Build context string
        context_parts = []
        citations = []
        for idx, res in enumerate(retrieved_chunks):
            context_parts.append(f"[Source {idx+1} | Page {res.page_number}]:\n{res.text}")
            citations.append(Citation(
                page_number=res.page_number,
                chunk_id=res.chunk_id,
                relevance_score=res.score,
                excerpt=res.text[:120] + ("..." if len(res.text) > 120 else "")
            ))

        context_str = "\n\n---\n\n".join(context_parts)

        # Format chat history
        history_str = ""
        if chat_history:
            for turn in chat_history[-4:]:
                history_str += f"{turn.get('role', 'user').title()}: {turn.get('content', '')}\n"

        prompt = f"""You are a helpful and precise Learning Assistant.
Answer the user's question based ONLY on the provided document excerpts.

### INSTRUCTIONS:
1. Ground your answer in the text. Cite page numbers whenever stating a fact (e.g. '[Page 2]').
2. If the excerpts do not contain enough information to answer, state clearly: "Based on the provided document, I cannot find sufficient details to answer this question."
3. Do not invent or extrapolate beyond what is given.
4. Keep the explanation clear, educational, and well-structured with bullet points when appropriate.

### RECENT CHAT HISTORY:
{history_str if history_str else 'No previous context.'}

### RETRIEVED DOCUMENT CONTEXT:
{context_str}

### USER QUESTION:
{query}"""

        system_prompt = (
            "You are a factual academic AI assistant. Ground all statements in the provided context and cite page numbers."
        )

        try:
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                json_mode=False
            )
        except Exception as e:
            answer = f"Error processing query: {str(e)}. Please check your API key configuration."

        # Estimate confidence based on top search score
        top_score = retrieved_chunks[0].score if retrieved_chunks else 0.0
        confidence = "High" if top_score > 0.65 else ("Medium" if top_score > 0.4 else "Low")

        return ChatResponse(
            answer=answer.strip(),
            citations=citations,
            confidence_level=confidence
        )
