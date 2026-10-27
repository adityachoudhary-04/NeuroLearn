import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.llm_provider import BaseLLMProvider
from core.preprocessor import TextChunk

@dataclass
class Flashcard:
    id: str
    front: str
    back: str
    concept: str
    page_number: int
    doc_id: str


class FlashcardGenerator:
    """Generates study flashcards (Term/Question -> Concise Definition/Answer)."""

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def generate_from_chunks(
        self,
        chunks: List[TextChunk],
        num_cards: int = 6
    ) -> List[Flashcard]:
        if not chunks:
            return []

        sorted_chunks = sorted(chunks, key=lambda c: c.salience_score, reverse=True)
        selected_chunks = sorted_chunks[:max(3, min(len(sorted_chunks), num_cards * 2))]

        context_blocks = []
        for idx, chunk in enumerate(selected_chunks):
            context_blocks.append(
                f"[Passage {idx+1} | Page {chunk.page_number} | Concepts: {', '.join(chunk.key_concepts)}]\n{chunk.text}"
            )
        full_context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a master study coach.
From the following context, create exactly {num_cards} high-impact study flashcards.

### GUIDELINES:
1. Front: A concise question, key term, acronym, or concept to test recall.
2. Back: A clear, memorable explanation, formula, or key definition (2-3 sentences max).
3. Concept: The broader topic or category.
4. Reference the page number.

### DOCUMENT CONTEXT:
{full_context}

### OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
{{
  "flashcards": [
    {{
      "front": "What is ...?",
      "back": "Clear, concise definition or explanation.",
      "concept": "Topic Name",
      "page_number": 1
    }}
  ]
}}"""

        system_prompt = "You are an educational study coach. Output ONLY valid parseable JSON."

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                json_mode=True
            )
            raw_cards = self._parse_json_response(raw_response)
        except Exception as e:
            print(f"Error during flashcard generation: {e}")
            raw_cards = self._fallback_rule_based(selected_chunks, num_cards)

        doc_id = chunks[0].doc_id if chunks else "doc_unknown"
        card_objects: List[Flashcard] = []
        for i, item in enumerate(raw_cards):
            front = item.get("front", "").strip()
            back = item.get("back", "").strip()
            if not front or not back:
                continue

            card_obj = Flashcard(
                id=f"card_{doc_id}_{i+1}",
                front=front,
                back=back,
                concept=item.get("concept", "General Concept"),
                page_number=int(item.get("page_number", selected_chunks[0].page_number if selected_chunks else 1)),
                doc_id=doc_id
            )
            card_objects.append(card_obj)

        return card_objects[:num_cards]

    def _parse_json_response(self, text: str) -> List[Dict[str, Any]]:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "flashcards" in data:
                return data["flashcards"]
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

    def _fallback_rule_based(self, chunks: List[TextChunk], num_cards: int) -> List[Dict[str, Any]]:
        cards = []
        for idx, chunk in enumerate(chunks[:num_cards]):
            concept = chunk.key_concepts[0] if chunk.key_concepts else f"Concept {idx+1}"
            cards.append({
                "front": f"Define or explain: {concept.title()}",
                "back": chunk.text[:150] + "...",
                "concept": concept.title(),
                "page_number": chunk.page_number
            })
        return cards
