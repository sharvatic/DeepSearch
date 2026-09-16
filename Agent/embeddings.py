import hashlib
from typing import List, Sequence

import numpy as np


class SafeEmbeddings:
    """Ollama embeddings with a deterministic local fallback."""

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        try:
            from langchain_ollama import OllamaEmbeddings

            self._ollama_emb = OllamaEmbeddings(model=model_name)
        except Exception:
            self._ollama_emb = None

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        text_list = list(texts)
        if self._ollama_emb:
            try:
                return self._ollama_emb.embed_documents(text_list)
            except Exception:
                pass

        return [self._fallback_embedding(text) for text in text_list]

    @staticmethod
    def _fallback_embedding(text: str) -> List[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vector = rng.randn(768).astype(np.float32)
        norm = np.linalg.norm(vector)
        return (vector / norm).tolist() if norm > 0 else vector.tolist()


def get_embeddings() -> SafeEmbeddings:
    return SafeEmbeddings()
