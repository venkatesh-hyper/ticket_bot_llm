from sentence_transformers import SentenceTransformer

class Retriever:
    """Search ChromaDB for semantically relevant chunks."""

    def __init__(self, vector_store, model_name="all-MiniLM-L6-v2"):
        self.vector_store = vector_store
        self.embedder = SentenceTransformer(model_name)

    def get_relevant_chunks(self, query, top_k=4):
        """Return top_k chunks most similar to the query."""
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        results = self.vector_store.search(q_emb, top_k=top_k)
        contexts = []
        for doc, meta, dist in results:
            text = doc.strip().replace("\n", " ")
            page = meta.get("page", "N/A")
            contexts.append(f"[Page {page}] {text}")
        return contexts
