import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_dir="chroma_storage"):
        print(f" Initializing ChromaDB at {persist_dir}")
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        self.client = chromadb.Client(settings)
        self.collection = self.client.get_or_create_collection("tickets")

    def add_chunks(self, chunks, embeddings, metadatas):
        """Store chunks + embeddings."""
        ids = [f"doc_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            embeddings=[emb.tolist() for emb in embeddings],
            metadatas=metadatas,
            ids=ids
        )
        self.client.persist()
        print(f"✅ Stored {len(chunks)} chunks in Chroma.")

    def search(self, query_embedding, top_k=4):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        return list(zip(docs, metas, dists))
