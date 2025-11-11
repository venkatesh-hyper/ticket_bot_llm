import chromadb

class VectorStore:
    def __init__(self, persist_dir="chroma_storage"):
        print(f"Initializing ChromaDB at {persist_dir}")
        try:
            # Persistent client is the new standard
            self.client = chromadb.PersistentClient(path=persist_dir)
        except Exception as e:
            print(f" Persistent client failed ({e}), using Ephemeral fallback.")
            self.client = chromadb.EphemeralClient()

        # Create or get a collection
        self.collection = self.client.get_or_create_collection("tickets")

    def add_chunks(self, chunks, embeddings, metadatas):
        ids = [f"doc_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            embeddings=[emb.tolist() for emb in embeddings],
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Stored {len(chunks)} chunks in ChromaDB.")

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
