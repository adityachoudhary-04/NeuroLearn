import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.llm_provider import BaseLLMProvider
from core.preprocessor import TextChunk

@dataclass
class DocumentSummary:
    title: str
    executive_summary: str
    key_takeaways: List[str]
    core_concepts: List[Dict[str, str]]
    recommended_study_plan: List[str]
    total_pages_analyzed: int
    doc_id: str


class SummaryGenerator:
    """Generates structured summaries, key takeaways, and learning outlines."""

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def generate_from_chunks(
        self,
        chunks: List[TextChunk],
        doc_title: str = "Document"
    ) -> DocumentSummary:
        if not chunks:
            return DocumentSummary(
                title=doc_title,
                executive_summary="No content provided to summarize.",
                key_takeaways=[],
                core_concepts=[],
                recommended_study_plan=[],
                total_pages_analyzed=0,
                doc_id=""
            )

        # Select top salience chunks across distinct pages for broad coverage
        pages_seen = set()
        representative_chunks: List[TextChunk] = []
        for c in sorted(chunks, key=lambda x: x.salience_score, reverse=True):
            if c.page_number not in pages_seen or len(representative_chunks) < 8:
                representative_chunks.append(c)
                pages_seen.add(c.page_number)
            if len(representative_chunks) >= 12:
                break

        # Sort chronologically by page number
        representative_chunks.sort(key=lambda c: (c.page_number, c.start_char))

        context_parts = [
            f"[Page {c.page_number}]: {c.text}"
            for c in representative_chunks
        ]
        context_str = "\n\n".join(context_parts)

        prompt = f"""You are an expert academic research assistant.
Generate a comprehensive, structured summary and study guide for '{doc_title}'.

### DOCUMENT PASSAGES:
{context_str}

### OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
{{
  "executive_summary": "High-level 2-3 paragraph overview of the central themes and findings.",
  "key_takeaways": [
    "Key takeaway point 1",
    "Key takeaway point 2",
    "Key takeaway point 3",
    "Key takeaway point 4"
  ],
  "core_concepts": [
    {{"term": "Term Name", "description": "Concise explanation of this concept"}},
    {{"term": "Second Term", "description": "Concise explanation"}}
  ],
  "recommended_study_plan": [
    "Step 1: Focus on ...",
    "Step 2: Understand the difference between ...",
    "Step 3: Practice applying ..."
  ]
}}"""

        system_prompt = "You are an analytical academic synthesizer. Output ONLY valid parseable JSON."

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                json_mode=True
            )
            data = self._parse_json_response(raw_response)
        except Exception as e:
            print(f"Error during summary generation: {e}")
            data = self._fallback_summary(representative_chunks, doc_title)

        doc_id = chunks[0].doc_id if chunks else "doc_unknown"
        total_pages = max(c.page_number for c in chunks) if chunks else 1

        return DocumentSummary(
            title=doc_title,
            executive_summary=data.get("executive_summary", "Summary unavailable."),
            key_takeaways=data.get("key_takeaways", []),
            core_concepts=data.get("core_concepts", []),
            recommended_study_plan=data.get("recommended_study_plan", []),
            total_pages_analyzed=total_pages,
            doc_id=doc_id
        )

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {}

    def _fallback_summary(self, chunks: List[TextChunk], doc_title: str) -> Dict[str, Any]:
        first_few = " ".join([c.text[:200] for c in chunks[:3]])
        return {
            "executive_summary": f"This document '{doc_title}' covers key aspects including: {first_few}...",
            "key_takeaways": [
                f"Page {c.page_number} introduces {', '.join(c.key_concepts[:2])}"
                for c in chunks[:4] if c.key_concepts
            ],
            "core_concepts": [
                {"term": c.key_concepts[0].title(), "description": f"Covered on page {c.page_number}"}
                for c in chunks if c.key_concepts
            ][:5],
            "recommended_study_plan": [
                "Review the core definitions in the first sections",
                "Practice generated questions and flashcards",
                "Review key takeaways before testing"
            ]
        }
