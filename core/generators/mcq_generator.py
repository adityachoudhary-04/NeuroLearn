import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.llm_provider import BaseLLMProvider
from core.preprocessor import TextChunk

@dataclass
class MCQ:
    id: str
    question: str
    options: List[str]  # e.g., ["A) ...", "B) ...", "C) ...", "D) ..."]
    correct_answer: str  # "A", "B", "C", or "D"
    explanation: str
    difficulty: str
    blooms_level: str
    page_number: int
    doc_id: str


class MCQGenerator:
    """Generates 4-choice Multiple Choice Questions (quizzes) with explanations."""

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def generate_from_chunks(
        self,
        chunks: List[TextChunk],
        num_questions: int = 5,
        target_difficulty: Optional[str] = None
    ) -> List[MCQ]:
        if not chunks:
            return []

        sorted_chunks = sorted(chunks, key=lambda c: c.salience_score, reverse=True)
        selected_chunks = sorted_chunks[:max(3, min(len(sorted_chunks), num_questions * 2))]

        context_blocks = []
        for idx, chunk in enumerate(selected_chunks):
            context_blocks.append(
                f"[Source Passage {idx+1} | Page {chunk.page_number}]\n{chunk.text}"
            )
        full_context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are an educational examination creator.
Based ONLY on the following context, create exactly {num_questions} Multiple Choice Questions (MCQs).

### GUIDELINES:
1. Target Difficulty: {target_difficulty if target_difficulty else 'Balanced (Easy, Medium, Hard)'}
2. Each question MUST have exactly 4 choices labeled 'A)', 'B)', 'C)', 'D)'.
3. Only ONE choice must be strictly correct according to the context. The other three choices must be plausible distractors.
4. Provide a clear pedagogical explanation justifying the correct answer.
5. Reference the page number where the fact is stated.

### DOCUMENT CONTEXT:
{full_context}

### OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
{{
  "mcqs": [
    {{
      "question": "What is ...?",
      "options": [
        "A) First option text",
        "B) Second option text",
        "C) Third option text",
        "D) Fourth option text"
      ],
      "correct_answer": "A",
      "explanation": "Why choice A is correct based on the text.",
      "difficulty": "Easy" | "Medium" | "Hard",
      "blooms_level": "Remember" | "Understand" | "Apply" | "Analyze",
      "page_number": 1
    }}
  ]
}}"""

        system_prompt = "You are a test assessment AI. Output ONLY valid parseable JSON."

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                json_mode=True
            )
            raw_mcqs = self._parse_json_response(raw_response)
        except Exception as e:
            print(f"Error during MCQ generation: {e}")
            raw_mcqs = self._fallback_rule_based(selected_chunks, num_questions)

        doc_id = chunks[0].doc_id if chunks else "doc_unknown"
        mcq_objects: List[MCQ] = []
        for i, item in enumerate(raw_mcqs):
            q_text = item.get("question", "").strip()
            options = item.get("options", [])
            correct = item.get("correct_answer", "A").strip().upper()
            if not q_text or len(options) < 2:
                continue

            # Ensure correct answer is single character
            if correct.startswith("A"): correct = "A"
            elif correct.startswith("B"): correct = "B"
            elif correct.startswith("C"): correct = "C"
            elif correct.startswith("D"): correct = "D"
            else: correct = "A"

            mcq_obj = MCQ(
                id=f"mcq_{doc_id}_{i+1}",
                question=q_text,
                options=options,
                correct_answer=correct,
                explanation=item.get("explanation", "The answer is supported by the document text."),
                difficulty=item.get("difficulty", "Medium"),
                blooms_level=item.get("blooms_level", "Understand"),
                page_number=int(item.get("page_number", selected_chunks[0].page_number if selected_chunks else 1)),
                doc_id=doc_id
            )
            mcq_objects.append(mcq_obj)

        return mcq_objects[:num_questions]

    def _parse_json_response(self, text: str) -> List[Dict[str, Any]]:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "mcqs" in data:
                return data["mcqs"]
            elif isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return []

    def _fallback_rule_based(self, chunks: List[TextChunk], num_questions: int) -> List[Dict[str, Any]]:
        results = []
        for idx, chunk in enumerate(chunks[:num_questions]):
            results.append({
                "question": f"Which concept is discussed on page {chunk.page_number}?",
                "options": [
                    f"A) {chunk.key_concepts[0] if chunk.key_concepts else 'Primary methodology'}",
                    "B) Unrelated external framework",
                    "C) Hypothetical non-context concept",
                    "D) Deprecated procedure"
                ],
                "correct_answer": "A",
                "explanation": f"Page {chunk.page_number} focuses on the topic discussed in the passage.",
                "difficulty": "Easy",
                "blooms_level": "Remember",
                "page_number": chunk.page_number
            })
        return results
