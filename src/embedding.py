import numpy as np
import tqdm


class Embedding:
    def __init__(self, model=None, model_name: str = 'multi-qa-distilbert-cos-v1'):
        # Do not import or initialize the heavy SentenceTransformer at module import time.
        self.model = model
        self.model_name = model_name

    def _ensure_model(self):
        if self.model is None:
            # Local import to avoid importing transformers/torchvision during Streamlit startup
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)

    def create(self, content: str):
        self._ensure_model()
        return self.model.encode(content)

    def create_batch(self, chunks: list[dict]) -> np.array:
        self._ensure_model()
        embedded_chunks = []
        for chunk in tqdm.tqdm(chunks):
            embedding = self.create(content=chunk.get("content", ""))
            embedded_chunks.append(embedding)
        return np.array(embedded_chunks)
    