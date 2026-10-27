import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from core.extractor import DocumentExtract, PageContent

@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    start_char: int
    end_char: int
    key_concepts: List[str] = field(default_factory=list)
    salience_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextPreprocessor:
    """Handles text cleaning, recursive chunking, and concept/keyword extraction."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean raw PDF extracted text: fix hyphenated line breaks, collapse whitespaces."""
        if not text:
            return ""
        # Fix hyphenated words broken across lines (e.g., "infor- \nmation" -> "information")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Replace standalone newlines within sentences with spaces
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # Collapse multi-newlines into double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalize non-breaking spaces & tabs
        text = text.replace('\xa0', ' ').replace('\t', ' ')
        # Collapse multiple spaces
        text = re.sub(r'[ ]{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def chunk_document(
        extract: DocumentExtract,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        salience_method: str = "tfidf",
        embedding_provider: Any = None
    ) -> List[TextChunk]:
        """
        Chunks the document page-aware using recursive character splitting.
        Preserves page number provenance for citations.
        """
        chunks: List[TextChunk] = []
        global_chunk_idx = 0

        for page in extract.pages:
            cleaned_page = TextPreprocessor.clean_text(page.text)
            if not cleaned_page:
                continue

            page_chunks = TextPreprocessor._split_text(
                cleaned_page,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            current_pos = 0
            for p_idx, chunk_str in enumerate(page_chunks):
                if not chunk_str.strip():
                    continue

                chunk_id = f"{extract.document_id}_p{page.page_number}_c{p_idx+1}"
                start_char = current_pos
                end_char = current_pos + len(chunk_str)
                current_pos += max(1, len(chunk_str) - chunk_overlap)

                chunk_obj = TextChunk(
                    chunk_id=chunk_id,
                    doc_id=extract.document_id,
                    text=chunk_str,
                    page_number=page.page_number,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "doc_id": extract.document_id,
                        "title": extract.title,
                        "page_number": page.page_number,
                        "chunk_index": global_chunk_idx,
                        "source_type": extract.source_type
                    }
                )
                chunks.append(chunk_obj)
                global_chunk_idx += 1

        # Extract key concepts and calculate salience scores using chosen method
        TextPreprocessor.extract_concepts_and_salience(
            chunks,
            method=salience_method,
            embedding_provider=embedding_provider
        )

        return chunks

    @staticmethod
    def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Recursive chunk splitting prioritizing paragraphs, then sentences, then spaces."""
        if len(text) <= chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
        
        def split_helper(subtext: str, sep_idx: int) -> List[str]:
            if len(subtext) <= chunk_size or sep_idx >= len(separators):
                return [subtext] if subtext.strip() else []

            sep = separators[sep_idx]
            if sep == "":
                # Fallback hard cut
                return [subtext[i:i + chunk_size] for i in range(0, len(subtext), chunk_size - chunk_overlap)]

            splits = subtext.split(sep)
            result = []
            current = ""

            for part in splits:
                candidate = (current + sep + part).strip() if current else part.strip()
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    if len(part) > chunk_size:
                        result.extend(split_helper(part, sep_idx + 1))
                        current = ""
                    else:
                        current = part.strip()

            if current and current.strip():
                result.append(current.strip())

            return result

        raw_splits = split_helper(text, 0)
        
        # Merge small consecutive fragments and apply sliding overlap
        merged = []
        i = 0
        while i < len(raw_splits):
            chunk = raw_splits[i]
            if len(chunk) < chunk_size // 3 and i + 1 < len(raw_splits):
                chunk = f"{chunk} {raw_splits[i+1]}"
                i += 1
            merged.append(chunk.strip())
            i += 1

        return merged

    @staticmethod
    def extract_concepts_and_salience(
        chunks: List[TextChunk],
        top_n_keywords: int = 5,
        method: str = "tfidf",
        embedding_provider: Any = None
    ):
        """
        Calculates key concepts and salience scores for each chunk.
        Supported methods:
          - 'tfidf': Statistical Term Frequency - Inverse Document Frequency
          - 'textrank': Graph-Based PageRank on word co-occurrence network
          - 'embedding': Dense Vector Semantic Similarity (KeyBERT-style)
        """
        if not chunks:
            return

        method = (method or "tfidf").lower().strip()

        if method in ["textrank", "graph"]:
            TextRankConceptExtractor.extract(chunks, top_n_keywords=top_n_keywords)
        elif method in ["embedding", "keybert", "dense"]:
            EmbeddingConceptExtractor.extract(chunks, top_n_keywords=top_n_keywords, embedding_provider=embedding_provider)
        else:
            TFIDFConceptExtractor.extract(chunks, top_n_keywords=top_n_keywords)


class TFIDFConceptExtractor:
    """Statistical TF-IDF concept and salience extractor."""
    @staticmethod
    def extract(chunks: List[TextChunk], top_n_keywords: int = 5):
        texts = [c.text for c in chunks]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words="english",
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            for idx, chunk in enumerate(chunks):
                row = tfidf_matrix[idx].toarray()[0]
                top_indices = row.argsort()[::-1][:top_n_keywords]
                chunk_keywords = [feature_names[i] for i in top_indices if row[i] > 0]
                
                chunk.key_concepts = chunk_keywords
                chunk.salience_score = float(sum(row[top_indices])) if chunk_keywords else 0.1
                chunk.metadata["key_concepts"] = chunk_keywords
                chunk.metadata["salience_score"] = round(chunk.salience_score, 4)
                chunk.metadata["salience_method"] = "tfidf"

        except Exception:
            _fallback_concept_extraction(chunks, top_n_keywords, method="tfidf")


class TextRankConceptExtractor:
    """
    Graph-based concept extractor implementing TextRank (Mihalcea & Tarau).
    Builds a word co-occurrence graph within sliding window W and runs PageRank power iteration.
    """
    @staticmethod
    def extract(chunks: List[TextChunk], top_n_keywords: int = 5, window_size: int = 3):
        import numpy as np

        stopwords = {
            "the", "and", "for", "with", "from", "into", "that", "this", "are", "was",
            "were", "been", "have", "has", "had", "they", "them", "their", "will", "would",
            "could", "should", "than", "then", "which", "what", "where", "when", "also",
            "more", "some", "such", "only", "other", "about", "these", "those", "both"
        }

        for chunk in chunks:
            raw_words = re.findall(r'\b[A-Za-z]{3,}\b', chunk.text.lower())
            tokens = [w for w in raw_words if w not in stopwords]

            if len(tokens) < 3:
                _fallback_single_chunk(chunk, top_n_keywords, "textrank")
                continue

            vocab = list(dict.fromkeys(tokens))
            word_to_id = {w: i for i, w in enumerate(vocab)}
            N = len(vocab)

            # Build adjacency matrix
            A = np.zeros((N, N), dtype=float)
            for i in range(len(tokens)):
                curr_id = word_to_id[tokens[i]]
                end_win = min(i + window_size, len(tokens))
                for j in range(i + 1, end_win):
                    neighbor_id = word_to_id[tokens[j]]
                    if curr_id != neighbor_id:
                        A[curr_id, neighbor_id] += 1.0
                        A[neighbor_id, curr_id] += 1.0

            # PageRank power iteration with damping factor d = 0.85
            row_sums = A.sum(axis=1, keepdims=True)
            M = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums != 0)
            
            d = 0.85
            pr = np.ones(N) / N
            for _ in range(30):
                pr = (1.0 - d) / N + d * M.T.dot(pr)

            # Sort tokens by PageRank centrality
            ranked_indices = pr.argsort()[::-1]
            top_words_set = set(vocab[i] for i in ranked_indices[:top_n_keywords * 2])

            # Compound phrase reconstruction (adjacent high-scoring words)
            phrases = []
            curr_phrase = []
            for w in tokens:
                if w in top_words_set:
                    curr_phrase.append(w)
                else:
                    if curr_phrase:
                        phrases.append(" ".join(curr_phrase))
                        curr_phrase = []
            if curr_phrase:
                phrases.append(" ".join(curr_phrase))

            # Deduplicate phrases while preserving rank
            unique_phrases = []
            for p in phrases:
                if p not in unique_phrases and len(p.split()) <= 3:
                    unique_phrases.append(p)

            # If no compound phrases, use top individual words
            if not unique_phrases:
                unique_phrases = [vocab[i] for i in ranked_indices[:top_n_keywords]]

            final_keywords = unique_phrases[:top_n_keywords]
            top_pr_sum = float(sum(pr[ranked_indices[:top_n_keywords]]))

            chunk.key_concepts = final_keywords
            chunk.salience_score = round(top_pr_sum * 10.0, 4)
            chunk.metadata["key_concepts"] = final_keywords
            chunk.metadata["salience_score"] = chunk.salience_score
            chunk.metadata["salience_method"] = "textrank"


class EmbeddingConceptExtractor:
    """
    Embedding-based concept extractor (KeyBERT-style).
    Extracts candidate unigrams and bigrams, computes cosine similarity
    against the full chunk vector, and selects semantically representative concepts.
    """
    @staticmethod
    def extract(chunks: List[TextChunk], top_n_keywords: int = 5, embedding_provider: Any = None):
        import numpy as np

        if embedding_provider is None:
            from core.embedding_provider import get_embedding_provider
            embedding_provider = get_embedding_provider()

        stopwords = {
            "the", "and", "for", "with", "from", "into", "that", "this", "are", "was",
            "were", "been", "have", "has", "had", "they", "them", "their", "will", "would",
            "could", "should", "than", "then", "which", "what", "where", "when", "also",
            "more", "some", "such", "only", "other", "about", "these", "those", "both",
            "can", "not", "over", "each", "most", "just"
        }

        for chunk in chunks:
            # 1. Extract candidate n-grams (1-gram and 2-gram)
            words = re.findall(r'\b[A-Za-z]{3,}\b', chunk.text.lower())
            candidates = set()
            for i in range(len(words)):
                if words[i] not in stopwords:
                    candidates.add(words[i])
                    if i + 1 < len(words) and words[i + 1] not in stopwords:
                        candidates.add(f"{words[i]} {words[i+1]}")

            cand_list = list(candidates)[:25]
            if not cand_list:
                _fallback_single_chunk(chunk, top_n_keywords, "embedding")
                continue

            try:
                # 2. Embed chunk document and candidates
                doc_vec = np.array(embedding_provider.embed_text(chunk.text))
                cand_vecs = np.array(embedding_provider.embed_batch(cand_list))

                # 3. Compute cosine similarities
                doc_norm = np.linalg.norm(doc_vec)
                cand_norms = np.linalg.norm(cand_vecs, axis=1)
                sims = np.dot(cand_vecs, doc_vec) / (cand_norms * doc_norm + 1e-9)

                # 4. Rank candidates by similarity
                scored_cands = sorted(zip(cand_list, sims), key=lambda x: x[1], reverse=True)
                
                # Filter partial substrings for diversity
                selected = []
                for term, score in scored_cands:
                    if not any(term in s or s in term for s in selected):
                        selected.append(term)
                    if len(selected) >= top_n_keywords:
                        break

                if not selected:
                    selected = [c[0] for c in scored_cands[:top_n_keywords]]

                avg_sim = float(np.mean([s for _, s in scored_cands[:top_n_keywords]]))

                chunk.key_concepts = selected
                chunk.salience_score = round(max(0.1, avg_sim), 4)
                chunk.metadata["key_concepts"] = selected
                chunk.metadata["salience_score"] = chunk.salience_score
                chunk.metadata["salience_method"] = "embedding"

            except Exception as e:
                print(f"Warning: Embedding concept extraction failed ({e}), falling back.")
                _fallback_single_chunk(chunk, top_n_keywords, "embedding")


def _fallback_single_chunk(chunk: TextChunk, top_n_keywords: int, method: str):
    words = re.findall(r'\b[A-Za-z]{4,}\b', chunk.text.lower())
    stop = {"with", "that", "this", "from", "they", "will", "have", "more", "their", "about", "which"}
    filtered = [w for w in words if w not in stop]
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n_keywords]
    chunk.key_concepts = [w[0] for w in sorted_words]
    chunk.salience_score = float(len(sorted_words) / 10.0)
    chunk.metadata["key_concepts"] = chunk.key_concepts
    chunk.metadata["salience_score"] = round(chunk.salience_score, 4)
    chunk.metadata["salience_method"] = method


def _fallback_concept_extraction(chunks: List[TextChunk], top_n_keywords: int, method: str):
    for c in chunks:
        _fallback_single_chunk(c, top_n_keywords, method)

