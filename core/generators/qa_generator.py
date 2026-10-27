import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from core.llm_provider import BaseLLMProvider
from core.preprocessor import TextChunk

@dataclass
class QAPair:
    id: str
    question: str
    answer: str
    difficulty: str  # "Easy", "Medium", "Hard"
    blooms_level: str  # "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
    context_quote: str
    page_number: int
    doc_id: str
    salience_score: float = 0.0


class QAGenerator:
    """Generates Bloom's Taxonomy-classified Question-Answer pairs with deduplication."""

    BLOOMS_TAXONOMY = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    DIFFICULTIES = ["Easy", "Medium", "Hard"]

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def generate_from_chunks(
        self,
        chunks: List[TextChunk],
        num_questions: int = 5,
        target_difficulty: Optional[str] = None,
        target_bloom: Optional[str] = None
    ) -> List[QAPair]:
        if not chunks:
            return []

        # Sort chunks by salience to pick the most informative passages
        sorted_chunks = sorted(chunks, key=lambda c: c.salience_score, reverse=True)
        selected_chunks = sorted_chunks[:max(3, min(len(sorted_chunks), num_questions * 2))]

        context_blocks = []
        for idx, chunk in enumerate(selected_chunks):
            concepts = ", ".join(chunk.key_concepts) if chunk.key_concepts else "N/A"
            context_blocks.append(
                f"[Source Chunk {idx+1} | Page {chunk.page_number} | Concepts: {concepts}]\n{chunk.text}"
            )
        full_context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are an expert educational assessment specialist and pedagogical AI.
Analyze the provided document context and generate exactly {num_questions} high-quality Question & Answer pairs.

### GUIDELINES:
1. Target Difficulty: {target_difficulty if target_difficulty else 'Balanced mix of Easy, Medium, Hard'}
2. Target Bloom's Taxonomy: {target_bloom if target_bloom else 'Spread across: Remember, Understand, Apply, Analyze, Evaluate, Create'}
3. Every question must be self-contained, grammatically clear, and directly answered by the context.
4. Ground every answer with a direct verbatim quote or evidence from the text.
5. Identify the exact Page Number where the answer is found.

### DOCUMENT CONTEXT:
{full_context}

### OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
{{
  "qa_pairs": [
    {{
      "question": "Clear, specific question?",
      "answer": "Accurate, comprehensive answer based solely on the text.",
      "difficulty": "Easy" | "Medium" | "Hard",
      "blooms_level": "Remember" | "Understand" | "Apply" | "Analyze" | "Evaluate" | "Create",
      "context_quote": "Exact short sentence from context supporting the answer",
      "page_number": 1
    }}
  ]
}}"""

        system_prompt = (
            "You are a strict academic evaluator. Output ONLY valid parseable JSON. "
            "Never hallucinate outside the given text."
        )

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                json_mode=True
            )
            raw_pairs = self._parse_json_response(raw_response)
        except Exception as e:
            print(f"Error during LLM QA generation: {e}")
            raw_pairs = self._fallback_rule_based(selected_chunks, num_questions)

        # Convert to QAPair dataclasses and assign doc_id
        doc_id = chunks[0].doc_id if chunks else "doc_unknown"
        qa_objects: List[QAPair] = []
        for i, item in enumerate(raw_pairs):
            q_text = item.get("question", "").strip()
            a_text = item.get("answer", "").strip()
            if not q_text or not a_text:
                continue

            qa_obj = QAPair(
                id=f"qa_{doc_id}_{i+1}",
                question=q_text,
                answer=a_text,
                difficulty=item.get("difficulty", "Medium"),
                blooms_level=item.get("blooms_level", "Understand"),
                context_quote=item.get("context_quote", ""),
                page_number=int(item.get("page_number", selected_chunks[0].page_number if selected_chunks else 1)),
                doc_id=doc_id,
                salience_score=selected_chunks[0].salience_score if selected_chunks else 0.5
            )
            qa_objects.append(qa_obj)

        # Deduplication step
        deduplicated = self._deduplicate_questions(qa_objects)
        return deduplicated[:num_questions]

    def _parse_json_response(self, text: str) -> List[Dict[str, Any]]:
        """Safely extracts JSON list of qa_pairs from text."""
        text = text.strip()
        # Remove code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "qa_pairs" in data:
                return data["qa_pairs"]
            elif isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            # Regex pattern fallback for JSON object
            match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return []

    def _deduplicate_questions(self, qa_list: List[QAPair], jaccard_threshold: float = 0.65) -> List[QAPair]:
        """Removes questions that share too much vocabulary or phrasing overlap."""
        unique_qas: List[QAPair] = []

        def tokenize(q: str):
            words = re.findall(r'\b\w+\b', q.lower())
            stop = {"what", "is", "the", "are", "how", "why", "which", "and", "in", "of", "to", "a", "an"}
            return set(w for w in words if w not in stop)

        for qa in qa_list:
            q_tokens = tokenize(qa.question)
            if not q_tokens:
                unique_qas.append(qa)
                continue

            is_duplicate = False
            for existing in unique_qas:
                e_tokens = tokenize(existing.question)
                intersection = len(q_tokens & e_tokens)
                union = len(q_tokens | e_tokens)
                similarity = intersection / (union + 1e-9)

                if similarity >= jaccard_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_qas.append(qa)

        return unique_qas

    def _fallback_rule_based(self, chunks: List[TextChunk], num_questions: int) -> List[Dict[str, Any]]:
        """Rule-based question generation if LLM is unavailable or fails."""
        results = []
        for idx, chunk in enumerate(chunks[:num_questions]):
            sentences = [s.strip() for s in re.split(r'\. |\n', chunk.text) if len(s.strip()) > 30]
            if sentences:
                s = sentences[0]
                results.append({
                    "question": f"According to page {chunk.page_number}, what is significant about: '{s[:60]}...'?",
                    "answer": s,
                    "difficulty": "Easy",
                    "blooms_level": "Remember",
                    "context_quote": s[:100],
                    "page_number": chunk.page_number
                })
        return results
