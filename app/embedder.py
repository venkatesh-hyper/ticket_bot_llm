from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=800, overlap=100):
        print(f"🔧 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        """Split text into overlapping chunks for embeddings."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = max(end - self.overlap, end)
        return chunks

    def embed_batch(self, texts):
        """Return list of vector embeddings."""
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
