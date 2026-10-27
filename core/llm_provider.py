import os
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from core.config import settings

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        pass


class GeminiLLMProvider(BaseLLMProvider):
    """Free tier LLM provider using Google Gemini (gemini-flash-latest, gemini-3.5-flash, gemini-3.8-flash)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-flash-latest"):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        if "gemini-2.5-flash" in model or "gemini-1.5-flash" in model:
            model = "gemini-flash-latest"
        self.model = model
        self._client = None
        if self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        if not self._client:
            raise ValueError("Gemini API key is missing. Set it in settings or the web UI.")
            
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
            response_mime_type="application/json" if json_mode else "text/plain"
        )
        
        candidates = [
            self.model,
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.8-flash",
            "gemini-flash-latest"
        ]
        # Deduplicate while preserving order
        unique_candidates = []
        for c in candidates:
            if c not in unique_candidates:
                unique_candidates.append(c)

        last_error = None
        for m in unique_candidates:
            try:
                response = self._client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=config
                )
                self.model = m
                return response.text or ""
            except Exception as e:
                last_error = e
                err_str = str(e).upper()
                if any(code in err_str for code in ["404", "NOT_FOUND", "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                    continue
                raise e

        raise last_error or RuntimeError("Gemini model generation failed.")


class GroqLLMProvider(BaseLLMProvider):
    """Free tier ultra-fast LLM provider using Groq Cloud (llama-3.3-70b-versatile, llama-3.1-8b-instant)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.model = model
        self._client = None
        if self.api_key:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        if not self._client:
            raise ValueError("Groq API key is missing. Set it in settings or the web UI.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        chat_completion = self._client.chat.completions.create(**kwargs)
        return chat_completion.choices[0].message.content or ""


class OpenAILLMProvider(BaseLLMProvider):
    """LLM provider using OpenAI (gpt-4o-mini, gpt-4o, etc.)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self._client = None
        if self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        if not self._client:
            raise ValueError("OpenAI API key is missing. Set it in settings or the web UI.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class ClaudeLLMProvider(BaseLLMProvider):
    """LLM provider using Anthropic Claude (claude-3-5-sonnet-20241022, claude-3-5-haiku)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model

    @property
    def provider_name(self) -> str:
        return "claude"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        if not self.api_key:
            raise ValueError("Anthropic API key is missing. Set it in settings or the web UI.")
        
        import httpx
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = httpx.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        return text


class MockLLMProvider(BaseLLMProvider):
    """
    Offline fallback simulator that generates realistic mock learning artifacts.
    Enables instant UI testing without requiring immediate API keys.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-simulator"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        # Check type of request
        prompt_lower = prompt.lower()
        if "summary" in prompt_lower or "executive" in prompt_lower or "takeaways" in prompt_lower:
            mock_summary = {
                "executive_summary": "This document outlines the architecture and principles of Retrieval-Augmented Generation (RAG) combined with Bloom's Taxonomy. It details how vector search and chunk salience scoring enable robust, pedagogical Q&A generation without hallucinations.",
                "key_takeaways": [
                    "RAG bridges external knowledge bases with dense LLM generation to guarantee factual accuracy.",
                    "Bloom's Taxonomy categorizes questions across six cognitive depth tiers from Remember to Create.",
                    "Recursive chunking preserving page metadata ensures strict citation provenance for all answers.",
                    "Vector embeddings and similarity search provide sub-second retrieval over large PDF collections."
                ],
                "core_concepts": [
                    {"term": "RAG", "description": "Retrieval-Augmented Generation framework grounding LLM outputs in external documents."},
                    {"term": "Bloom's Taxonomy", "description": "Hierarchical classification of learning objectives from recall to synthesis."},
                    {"term": "Salience Scoring", "description": "Information-density metric identifying highest-priority chunks for assessment generation."}
                ],
                "recommended_study_plan": [
                    "Review foundational RAG architecture concepts and chunking workflows",
                    "Understand how Bloom's cognitive tiers affect prompt design",
                    "Practice test questions and review flashcards"
                ]
            }
            if json_mode:
                return json.dumps(mock_summary)
            return mock_summary["executive_summary"]

        if "question" in prompt_lower or "qa" in prompt_lower or "quiz" in prompt_lower or "mcq" in prompt_lower or "flashcard" in prompt_lower:
            mock_data = {
                "qa_pairs": [
                    {
                        "question": "What is the primary objective of this document?",
                        "answer": "The primary objective is to outline the foundational principles and technical architecture described in the provided context.",
                        "difficulty": "Easy",
                        "blooms_level": "Understand",
                        "context_quote": "Extracted from foundational principles section.",
                        "page_number": 1
                    },
                    {
                        "question": "How does the proposed approach improve pipeline efficiency?",
                        "answer": "By implementing recursive chunking, selective embedding retrieval, and Bloom's taxonomy verification, the pipeline ensures high question relevance and reduces hallucination.",
                        "difficulty": "Medium",
                        "blooms_level": "Analyze",
                        "context_quote": "Extracted from technical evaluation section.",
                        "page_number": 1
                    }
                ],
                "mcqs": [
                    {
                        "question": "Which of the following best describes the RAG pipeline discussed?",
                        "options": [
                            "A) Text Extraction -> Chunking -> Vector DB -> Retrieval -> LLM Generation",
                            "B) Direct LLM generation without any document retrieval",
                            "C) Manual question creation only",
                            "D) None of the above"
                        ],
                        "correct_answer": "A",
                        "explanation": "A RAG pipeline extracts text, chunks it, generates embeddings into a vector store, and retrieves relevant context for the LLM.",
                        "difficulty": "Easy",
                        "blooms_level": "Remember"
                    }
                ],
                "flashcards": [
                    {
                        "front": "RAG (Retrieval-Augmented Generation)",
                        "back": "An AI framework that retrieves relevant facts from an external knowledge base (e.g. PDFs) to ground LLM generation.",
                        "concept": "Architecture"
                    },
                    {
                        "front": "Bloom's Taxonomy",
                        "back": "A hierarchical ordering of cognitive skills: Remember, Understand, Apply, Analyze, Evaluate, Create.",
                        "concept": "Pedagogy"
                    }
                ]
            }
            if json_mode:
                return json.dumps(mock_data)
            return "Mock generation completed successfully. Configure your GEMINI_API_KEY or GROQ_API_KEY in the settings panel for live AI generation."
        
        if json_mode:
            return json.dumps({"status": "ok", "message": "Demo response from Mock LLM Provider."})
        return "This is a response generated by the local offline assistant simulator. Add your free Gemini or Groq API key in the Settings tab to activate full live generation."


def get_llm_provider(
    name: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> BaseLLMProvider:
    """Factory to retrieve appropriate LLM provider with graceful fallbacks."""
    provider_name = (name or settings.DEFAULT_LLM_PROVIDER).lower()

    # An explicit mock/offline selection must win even when API keys exist in
    # .env.  This is used by local development, tests, and the UI's offline
    # mode.  Previously the factory fell through and silently selected Gemini.
    if provider_name in ["mock", "offline", "local"]:
        return MockLLMProvider()
    
    # 1. Gemini
    if provider_name == "gemini":
        key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if key:
            return GeminiLLMProvider(api_key=key, model=model or settings.DEFAULT_LLM_MODEL)
            
    # 2. Groq
    elif provider_name == "groq":
        key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if key:
            return GroqLLMProvider(api_key=key, model=model or "llama-3.3-70b-versatile")

    # 3. OpenAI
    elif provider_name == "openai":
        key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if key:
            return OpenAILLMProvider(api_key=key, model=model or "gpt-4o-mini")

    # 4. Claude
    elif provider_name == "claude":
        key = api_key or settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
        if key:
            return ClaudeLLMProvider(api_key=key, model=model or "claude-3-5-haiku-20241022")

    # If an unknown/default provider was requested, use any configured key
    # before falling back to the offline simulator.
    if settings.GEMINI_API_KEY:
        return GeminiLLMProvider(api_key=settings.GEMINI_API_KEY)
    if settings.GROQ_API_KEY:
        return GroqLLMProvider(api_key=settings.GROQ_API_KEY)
    if settings.OPENAI_API_KEY:
        return OpenAILLMProvider(api_key=settings.OPENAI_API_KEY)

    # Offline/Mock provider fallback
    return MockLLMProvider()
