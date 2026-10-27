"""Custom grounded test-paper generation using the existing LLM and chunks."""

import json
import re
from collections import Counter
from itertools import cycle
from typing import Any, Dict, List
from core.generators.qa_generator import QAGenerator, QAPair
from core.generators.mcq_generator import MCQGenerator, MCQ


SUPPORTED_TYPES = {
    "mcq", "true_false", "fill_blank", "very_short", "short", "descriptive", "numerical"
}


def _parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group(0)) if match else {}


def _sentences(chunk: Any) -> List[str]:
    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", ""))).strip()
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)
                 if 35 <= len(sentence.strip()) <= 320]
    if sentences:
        return sentences
    # PDF extraction often loses punctuation. Split an overlong page into
    # readable, word-boundary excerpts rather than showing a raw page dump.
    words = text.split()
    return [" ".join(words[start:start + 42]) for start in range(0, len(words), 42)
            if len(" ".join(words[start:start + 42])) >= 35]


def _concept(chunk: Any, sentence: str) -> str:
    concepts = getattr(chunk, "key_concepts", None) or []
    ignored = {"select", "using", "based", "given", "other", "also", "when", "this", "that", "there", "which", "should", "where", "their", "result", "could", "would", "about", "source", "output", "value", "values", "first", "second", "therefore", "hence", "step", "marks", "page", "chapter"}
    for item in concepts:
        candidate = re.sub(r"\s+", " ", str(item)).strip()
        words = re.findall(r"[A-Za-z]{2,}", candidate)
        # Reject formula fragments such as "yi wxi" and OCR artefacts. They
        # make a question technically grounded but meaningless to a student.
        if (candidate and candidate.lower() not in ignored and words
                and any(len(word) >= 4 for word in words)
                and not all(len(word) <= 3 for word in words)):
            return candidate
    words = re.findall(r"\b[A-Za-z][A-Za-z-]{4,}\b", sentence)
    preferred = {"regression", "classification", "residual", "objective", "algorithm", "probability", "distribution", "function", "method", "model", "equation", "principle", "process", "relationship", "accuracy", "entropy", "information", "decision", "learning", "feature", "variable"}
    return next((word for word in words if word.lower() in preferred), next((word for word in words if word.lower() not in ignored), "the source concept"))


def _fallback_question(kind: str, chunk: Any, ordinal: int, marks: Dict[str, float]) -> Dict[str, Any]:
    sentences = _sentences(chunk)
    sentence = sentences[ordinal % len(sentences)] if sentences else str(getattr(chunk, "text", "")).strip()[:240]
    concept = _concept(chunk, sentence)
    page = getattr(chunk, "page_number", 1)
    base = {
        "type": kind,
        "options": [],
        "explanation": f"This answer is supported by the source passage on page {page}.",
        "page_number": page,
        "marks": float(marks.get(kind, 1) or 1),
        "rubric": "",
    }
    if kind == "mcq":
        base.update({
            "question": f"According to the source, which statement correctly describes {concept}?",
            "options": [
                f"A) {sentence}",
                "B) The source states that this concept has no effect on the outcome.",
                "C) The source defines this concept as the opposite of the stated relationship.",
                "D) The source says this concept is not discussed in the material.",
            ],
            "answer": "A",
        })
    elif kind == "true_false":
        base.update({"question": f"True or False: {sentence}", "answer": "True"})
    elif kind == "fill_blank":
        words = re.findall(r"\b[A-Za-z][A-Za-z-]{4,}\b", sentence)
        term = concept if concept.lower() in sentence.lower() else (words[0] if words else "concept")
        blanked = re.sub(re.escape(term), "_____", sentence, count=1, flags=re.I)
        base.update({"question": f"Fill in the blank: {blanked}", "answer": term})
    elif kind == "very_short":
        base.update({"question": f"What does the source state about {concept}?", "answer": sentence})
    elif kind == "short":
        base.update({"question": f"Briefly explain {concept} using the source material.", "answer": sentence})
    elif kind == "descriptive":
        base.update({
            "question": f"Explain the role and significance of {concept} using evidence from the source.",
            "answer": sentence,
            "marks": float(marks.get(kind, 5) or 5),
            "rubric": "Award marks for identifying the concept, accurately explaining its role, and connecting the answer to the supplied source evidence.",
        })
    else:  # numerical
        numeric = next((candidate for candidate in sentences if re.search(r"\d", candidate)), sentence)
        base.update({
            "question": f"State the numerical value, probability, threshold, or quantity given in the source for {concept}.",
            "answer": numeric,
        })
    return base


def _fallback_paper(chunks: List[Any], counts: Dict[str, int], marks: Dict[str, float]) -> List[Dict[str, Any]]:
    source_chunks = [chunk for chunk in chunks if _sentences(chunk)] or chunks
    if not source_chunks:
        return []
    questions: List[Dict[str, Any]] = []
    chunk_cycle = cycle(source_chunks)
    ordinal = 0
    # Preserve the user's selected type distribution exactly.
    for kind, count in counts.items():
        for _ in range(count):
            questions.append(_fallback_question(kind, next(chunk_cycle), ordinal, marks))
            ordinal += 1
    return questions


def _valid_generated_questions(questions: Any, expected: Dict[str, int]) -> bool:
    if not isinstance(questions, list) or len(questions) != sum(expected.values()):
        return False
    seen = Counter()
    for question in questions:
        if not isinstance(question, dict):
            return False
        kind = str(question.get("type", ""))
        if kind not in expected or not str(question.get("question", "")).strip():
            return False
        if kind == "mcq" and (not isinstance(question.get("options"), list) or len(question["options"]) != 4):
            return False
        seen[kind] += 1
    return seen == Counter(expected)


def _blank_term(text: str) -> str:
    ignored = {"this", "that", "with", "from", "into", "which", "where", "there", "their", "about", "source", "according"}
    words = re.findall(r"\b[A-Za-z][A-Za-z-]{4,}\b", text)
    return next((word for word in words if word.lower() not in ignored), "concept")


def _question_from_qa(qa: QAPair, kind: str, marks: Dict[str, float]) -> Dict[str, Any]:
    evidence = (qa.context_quote or qa.answer).strip()
    base = {
        "type": kind,
        "options": [],
        "answer": qa.answer,
        "explanation": f"Grounded in the source evidence on page {qa.page_number}.",
        "page_number": qa.page_number,
        "marks": float(marks.get(kind, 1) or 1),
        "rubric": "",
    }
    if kind == "true_false":
        base.update({"question": f"True or False: {evidence}", "answer": "True"})
    elif kind == "fill_blank":
        term = _blank_term(evidence)
        base.update({"question": f"Fill in the blank: {re.sub(re.escape(term), '_____', evidence, count=1, flags=re.I)}", "answer": term})
    elif kind == "very_short":
        base["question"] = qa.question
        base["marks"] = float(marks.get(kind, 1) or 1)
    elif kind == "short":
        base["question"] = qa.question
        base["marks"] = float(marks.get(kind, 2) or 2)
    elif kind == "descriptive":
        base.update({
            "question": qa.question,
            "marks": float(marks.get(kind, 5) or 5),
            "rubric": "Award marks for accurate core concepts, factual correctness, completeness, and relevant source-grounded explanation.",
        })
    else:  # numerical
        numeric_evidence = evidence if re.search(r"\d", evidence) else qa.answer
        base.update({
            "question": f"Using the source material, state the relevant value or relationship: {qa.question}",
            "answer": numeric_evidence,
            "marks": float(marks.get(kind, 3) or 3),
        })
    return base


def _question_from_mcq(mcq: MCQ, marks: Dict[str, float]) -> Dict[str, Any]:
    return {
        "type": "mcq",
        "question": mcq.question,
        "options": mcq.options[:4],
        "answer": mcq.correct_answer,
        "explanation": mcq.explanation,
        "page_number": mcq.page_number,
        "marks": float(marks.get("mcq", 1) or 1),
        "rubric": "",
    }


def _generate_in_batches(generator, chunks: List[Any], total: int, kind: str, difficulty: str) -> List[Any]:
    """Use existing focused generators in small source windows for better coverage."""
    if total <= 0 or not chunks:
        return []
    results: List[Any] = []
    window_size, batch_size, offset = 20, 8, 0
    while len(results) < total and offset < max(len(chunks), 1) * 2:
        subset = chunks[offset:offset + window_size]
        if not subset:
            subset = chunks[:window_size]
        requested = min(batch_size, total - len(results))
        try:
            if kind == "mcq":
                batch = generator.generate_from_chunks(subset, requested, difficulty if difficulty != "Mixed" else None)
            else:
                batch = generator.generate_from_chunks(subset, requested, difficulty if difficulty != "Mixed" else None, None)
            for item in batch:
                question = getattr(item, "question", "").strip().lower()
                if question and question not in {getattr(existing, "question", "").strip().lower() for existing in results}:
                    results.append(item)
        except Exception:
            pass
        offset += window_size
    return results[:total]


def _source_tokens(chunks: List[Any]) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z]{4,}\b", " ".join(str(getattr(chunk, "text", "")) for chunk in chunks).lower()))


def _is_grounded(text: str, source_tokens: set[str]) -> bool:
    ignored = {"this", "that", "with", "from", "into", "which", "what", "when", "where", "their", "document", "source", "answer", "primary", "using", "based", "context"}
    tokens = {word for word in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()) if word not in ignored}
    return len(tokens & source_tokens) >= min(2, max(1, len(tokens)))


def _usable_qa(qa: QAPair, source_tokens: set[str]) -> bool:
    generic = (qa.question or "").lower().startswith("according to page")
    mock_quote = (qa.context_quote or "").lower().startswith("extracted from")
    return not generic and not mock_quote and _is_grounded(f"{qa.question} {qa.answer}", source_tokens)


def _usable_mcq(mcq: MCQ, source_tokens: set[str]) -> bool:
    generic = (mcq.question or "").lower().startswith("which concept is discussed")
    options = [str(option).strip() for option in (mcq.options or [])]
    # A technically valid MCQ is still unusable if every option is a generic
    # placeholder or if it does not have one labelled answer the student can select.
    has_labels = all(re.match(r"^[A-D][).]\\s+", option, re.I) for option in options)
    distinct_options = len({re.sub(r"^[A-D][).]\\s+", "", option, flags=re.I).lower() for option in options}) == 4
    return (
        not generic
        and len(options) == 4
        and has_labels
        and distinct_options
        and mcq.correct_answer in {"A", "B", "C", "D"}
        and _is_grounded(f"{mcq.question} {mcq.explanation} {' '.join(options)}", source_tokens)
    )


def _build_with_existing_generators(llm, chunks: List[Any], counts: Dict[str, int], marks: Dict[str, float], difficulty: str) -> List[Dict[str, Any]]:
    """Build quality paper items from the already-proven Q&A and MCQ pipelines."""
    fallback = _fallback_paper(chunks, counts, marks)
    fallback_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for item in fallback:
        fallback_by_type.setdefault(item["type"], []).append(item)

    mcqs = _generate_in_batches(MCQGenerator(llm), chunks, counts.get("mcq", 0), "mcq", difficulty)
    written_total = sum(count for kind, count in counts.items() if kind != "mcq")
    qas = _generate_in_batches(QAGenerator(llm), chunks, written_total, "qa", difficulty)
    tokens = _source_tokens(chunks)
    items: List[Dict[str, Any]] = []
    mcq_index = qa_index = 0
    seen_questions: set[str] = set()
    for kind, requested in counts.items():
        for _ in range(requested):
            if kind == "mcq" and mcq_index < len(mcqs):
                candidate = mcqs[mcq_index]
                mcq_index += 1
                normalized_question = re.sub(r"\W+", " ", candidate.question.lower()).strip()
                if normalized_question not in seen_questions and _usable_mcq(candidate, tokens):
                    items.append(_question_from_mcq(candidate, marks))
                    seen_questions.add(normalized_question)
                    continue
            elif kind != "mcq" and qa_index < len(qas):
                candidate = qas[qa_index]
                qa_index += 1
                normalized_question = re.sub(r"\W+", " ", candidate.question.lower()).strip()
                if normalized_question not in seen_questions and _usable_qa(candidate, tokens):
                    items.append(_question_from_qa(candidate, kind, marks))
                    seen_questions.add(normalized_question)
                    continue
            fallback_item = fallback_by_type[kind].pop(0)
            items.append(fallback_item)
            seen_questions.add(re.sub(r"\W+", " ", fallback_item["question"].lower()).strip())
    return items


def generate_test_paper(
    llm,
    chunks: List[Any],
    title: str,
    subject: str,
    question_counts: Dict[str, int],
    difficulty: str = "Mixed",
    duration_minutes: int = 60,
    marks: Dict[str, float] | None = None,
    include_answer_key: bool = True,
    include_explanations: bool = True,
    include_rubrics: bool = True,
    shuffle: bool = False,
    topic: str = "All topics",
) -> Dict[str, Any]:
    counts = {key: int(value) for key, value in question_counts.items() if int(value) > 0 and key in SUPPORTED_TYPES}
    total = sum(counts.values())
    maximum_marks = marks or {}
    selected = sorted(chunks, key=lambda item: getattr(item, "salience_score", 0), reverse=True)
    # Reuse the project's established MCQ and Q&A generation pipelines
    # instead of asking one oversized prompt to invent a mixed paper. This
    # preserves the quality controls those generators already apply.
    questions = _build_with_existing_generators(
        llm=llm,
        chunks=selected,
        counts=counts,
        marks=maximum_marks,
        difficulty=difficulty,
    )

    normalized = []
    for index, item in enumerate(questions[:total]):
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        kind = str(item.get("type") or "short")
        normalized.append({
            "id": f"paper_q_{index + 1}",
            "type": kind,
            "question": str(item.get("question", "")).strip(),
            "options": item.get("options", []) if isinstance(item.get("options", []), list) else [],
            "answer": str(item.get("answer", "")).strip(),
            "explanation": str(item.get("explanation", "")).strip() if include_explanations else "",
            "page_number": int(item.get("page_number", 1) or 1),
            "marks": float(item.get("marks", maximum_marks.get(kind, 1)) or 1),
            "rubric": str(item.get("rubric", "")).strip() if include_rubrics else "",
        })
    total_marks = round(sum(item["marks"] for item in normalized), 2)
    return {
        "title": title,
        "subject": subject,
        "duration_minutes": duration_minutes,
        "difficulty": difficulty,
        "questions": normalized,
        "total_questions": len(normalized),
        "total_marks": total_marks,
        "include_answer_key": include_answer_key,
        "shuffle": shuffle,
        "warning": None if len(normalized) >= total else f"Only {len(normalized)} source-grounded questions could be generated from this document (requested {total}).",
    }
