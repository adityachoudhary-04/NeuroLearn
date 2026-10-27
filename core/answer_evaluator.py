"""Semantic descriptive-answer evaluation built on the configured LLM."""

import json
import re
from typing import Any, Dict, Optional


def _clamp(value: Any, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = (text or "").strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _keywords(text: str) -> set[str]:
    """Small, transparent fallback normalizer for offline grading."""
    stop_words = {"that", "this", "with", "from", "into", "have", "been", "were", "will", "would", "should", "their", "there", "about", "which", "while", "where", "when", "because", "answer", "source"}
    result = set()
    for word in re.findall(r"\b[a-zA-Z]{4,}\b", (text or "").lower()):
        if word in stop_words:
            continue
        # This deliberately handles only common English inflections; it is a
        # fallback heuristic, while configured LLMs provide semantic grading.
        if word.endswith("ies"):
            word = word[:-3] + "y"
        elif word.endswith("ing") and len(word) > 6:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 5:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 5:
            word = word[:-1]
        result.add(word)
    return result


def evaluate_answer(
    llm,
    question: str,
    reference_answer: str,
    student_answer: str,
    source_context: str = "",
    max_marks: float = 10,
    rubric: str = "Assess conceptual correctness, factual accuracy, completeness, and meaningful reasoning."
) -> Dict[str, Any]:
    """Evaluate meaning, not string similarity, and always return bounded marks."""
    maximum = max(0.5, float(max_marks))
    prompt = f"""Evaluate a student's descriptive answer semantically.

Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}

Relevant source material:
{source_context or 'Not supplied'}

Marking rubric:
{rubric}

Accept equivalent wording, alternative valid examples, and different sentence structures.
Do not use exact string matching or embedding similarity alone. Award partial credit for
correct concepts, deduct for missing concepts and contradictions, and never reward facts
unsupported by the source. Return ONLY JSON:
{{
  "marks": 0,
  "max_marks": {maximum},
  "percentage": 0,
  "explanation": "short overall rationale",
  "strengths": ["what is correct"],
  "missing_concepts": ["what is missing"],
  "incorrect_points": ["incorrect or contradictory claims"],
  "improvement": "how to improve",
  "ideal_answer": "clear reference answer"
}}"""
    try:
        raw = llm.generate(
            prompt=prompt,
            system_prompt="You are a fair academic examiner. Evaluate meaning and concepts, not wording. Output valid JSON only.",
            temperature=0.1,
            json_mode=True,
        )
        result = _parse_json(raw)
    except Exception:
        result = None

    if not result or "marks" not in result:
        # Safe offline rubric: useful feedback without pretending to be an LLM.
        reference_terms = _keywords(reference_answer)
        student_terms = _keywords(student_answer)
        overlap = len(reference_terms & student_terms) / max(1, len(reference_terms))
        marks = maximum * _clamp(overlap, 0, 1)
        result = {
            "marks": marks,
            "max_marks": maximum,
            "percentage": marks / maximum * 100,
            "explanation": "Offline rubric estimate based on covered reference concepts; configure an LLM for semantic grading.",
            "strengths": ["Some reference concepts are present."] if marks else [],
            "missing_concepts": ["Compare your answer with the ideal answer and include each key concept."],
            "incorrect_points": [],
            "improvement": "Explain the key concepts in your own words and support them with document evidence.",
            "ideal_answer": reference_answer,
        }

    marks = _clamp(result.get("marks", 0), 0, maximum)
    result["marks"] = round(marks, 2)
    result["max_marks"] = maximum
    result["percentage"] = round(marks / maximum * 100, 1)
    for key in ("strengths", "missing_concepts", "incorrect_points"):
        value = result.get(key, [])
        result[key] = value if isinstance(value, list) else [str(value)]
    return result
