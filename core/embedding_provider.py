import os
import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from core.config import settings

class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Free tier embedding provider via Google Gemini (gemini-embedding-001, gemini-embedding-2)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-embedding-001"):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        # The global default is a local BERT model, so never pass that local
        # model name into Gemini's API when the provider is switched at runtime.
        # Gemini's Developer API accepts the cloud embedding model IDs below.
        model = str(model or "").strip()
        if model.startswith("models/"):
            model = model[7:]
        if not model or model in {"all-MiniLM-L6-v2", "bge-small-en-v1.5"}:
            model = "gemini-embedding-001"
        elif "text-embedding-004" in model:
            model = "gemini-embedding-001"
        self.model = model
        self._dim = 3072
        self._client = None

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to init Gemini genai.Client: {e}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        if not self._client:
            raise ValueError("Gemini API key is missing. Please provide it in settings or .env")
        
        for candidate_model in [self.model, "gemini-embedding-001", "gemini-embedding-2"]:
            try:
                response = self._client.models.embed_content(
                    model=candidate_model,
                    contents=text
                )
                emb = response.embeddings[0].values
                self._dim = len(emb)
                self.model = candidate_model
                return emb
            except Exception as e:
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    continue
                raise e
        raise RuntimeError("Failed to generate embedding with Gemini embedding models.")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._client:
            raise ValueError("Gemini API key is missing. Please provide it in settings or .env")
        if not texts:
            return []
        
        results: List[List[float]] = []
        batch_size = 30
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            success = False
            for candidate_model in [self.model, "gemini-embedding-001", "gemini-embedding-2"]:
                try:
                    response = self._client.models.embed_content(
                        model=candidate_model,
                        contents=chunk
                    )
                    for emb in response.embeddings:
                        results.append(emb.values)
                    self._dim = len(response.embeddings[0].values)
                    self.model = candidate_model
                    success = True
                    break
                except Exception as e:
                    if "404" in str(e) or "NOT_FOUND" in str(e):
                        continue
                    raise e
            if not success:
                raise RuntimeError("Failed to batch embed with Gemini models.")
        return results


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider via OpenAI (text-embedding-3-small, text-embedding-ada-002, etc.)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self._dim = 1536
        self._client = None

        if self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        if not self._client:
            raise ValueError("OpenAI API key is missing.")
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._client:
            raise ValueError("OpenAI API key is missing.")
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]


class LocalHashEmbeddingProvider(BaseEmbeddingProvider):
    """
    Offline/Fallback pseudo-dense embedding using TF-IDF and random projection (384-dim).
    Guarantees the system works even completely offline or with zero API keys.
    """
    
    def __init__(self, dim: int = 384):
        self._dim = dim
        # Seeded deterministic random projection matrix for consistent cosine similarity
        rng = np.random.RandomState(42)
        self.proj = rng.randn(1000, self._dim)
        self.proj /= np.linalg.norm(self.proj, axis=1, keepdims=True)

    @property
    def dimension(self) -> int:
        return self._dim

    def _text_to_vec(self, text: str) -> List[float]:
        tokens = text.lower().split()
        if not tokens:
            return [0.0] * self._dim
        
        vec = np.zeros(1000)
        for token in tokens:
            # Python's built-in hash is randomized for every process, which
            # made persisted Chroma vectors incompatible after a restart.
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % 1000
            vec[idx] += 1.0
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
            
        dense = np.dot(vec, self.proj)
        dense_norm = np.linalg.norm(dense)
        if dense_norm > 0:
            dense /= dense_norm
            
        return dense.tolist()

    def embed_text(self, text: str) -> List[float]:
        return self._text_to_vec(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vec(t) for t in texts]


class BertEmbeddingProvider(BaseEmbeddingProvider):
    """
    BERT / Sentence-BERT local embedding provider (e.g. all-MiniLM-L6-v2, bge-small-en-v1.5).
    Produces 384-dimensional dense semantic vectors locally on CPU/GPU without external API calls.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._dim = 384
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Startup must never block the whole web app while Hugging Face tries
            # to reach the network. Use an already cached model when available;
            # otherwise the deterministic local fallback below remains fully
            # functional and the user can still upload and query documents.
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
            dimension_getter = getattr(self._model, "get_embedding_dimension", None)
            if dimension_getter is None:
                dimension_getter = self._model.get_sentence_embedding_dimension
            self._dim = dimension_getter()
        except Exception:
            # Fallback if sentence-transformers/model files are unavailable.
            self._model = None

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def backend_name(self) -> str:
        """Expose whether SBERT is loaded or the provider's local safety fallback is used."""
        return "sentence-transformers" if self._model is not None else "local-hash-fallback"

    def embed_text(self, text: str) -> List[float]:
        if self._model is not None and hasattr(self._model, 'encode'):
            vec = self._model.encode(text)
            return vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        # Zero-dependency local deterministic semantic projection
        return LocalHashEmbeddingProvider(dim=self._dim).embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._model is not None and hasattr(self._model, 'encode'):
            vecs = self._model.encode(texts, batch_size=32)
            return vecs.tolist() if hasattr(vecs, 'tolist') else [list(v) for v in vecs]
        return LocalHashEmbeddingProvider(dim=self._dim).embed_batch(texts)


def get_embedding_provider(
    name: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> BaseEmbeddingProvider:
    """Factory to retrieve appropriate embedding provider."""
    provider_name = (name or settings.DEFAULT_EMBEDDING_PROVIDER).lower()
    
    if provider_name in ["bert", "sentence-transformers", "sbert", "huggingface"]:
        return BertEmbeddingProvider(model_name=model or "all-MiniLM-L6-v2")
        
    elif provider_name == "gemini":
        key = api_key or settings.GEMINI_API_KEY
        if key:
            try:
                # Do not use DEFAULT_EMBEDDING_MODEL here: that setting is the
                # local BERT model by default and is not a valid Gemini model.
                requested_model = model or "gemini-embedding-001"
                return GeminiEmbeddingProvider(api_key=key, model=requested_model)
            except Exception:
                pass
        # Cloud embeddings require a key. Prefer local BERT/SBERT before the
        # lightweight hash fallback so semantic retrieval still works offline.
        return BertEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        
    elif provider_name == "openai":
        key = api_key or settings.OPENAI_API_KEY
        if key:
            try:
                return OpenAIEmbeddingProvider(api_key=key, model=model or "text-embedding-3-small")
            except Exception:
                pass
        return BertEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        
    elif provider_name in ["local", "offline", "tfidf"]:
        return LocalHashEmbeddingProvider()
        
    # Unknown provider or provider initialization problem: use local BERT.
    return BertEmbeddingProvider(model_name="all-MiniLM-L6-v2")
