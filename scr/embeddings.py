from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(
                "Model loaded successfully. "
                f"Embedding dimensions: {self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, text: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not Loaded")
        embeddings = self.model.encode(text, show_progress_bar=True , normalize_embeddings=True)
        return embeddings

    def __call__(self, text: List[str]):
        return self.generate_embeddings(text)
