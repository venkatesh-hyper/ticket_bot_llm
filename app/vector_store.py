import os
import chromadb

class VectorStore:
    """Vector database wrapper with safe persistent setup and fallback."""

    def __init__(self, persist_dir="chroma_storage"):
        self.persist_dir = persist_dir
        print(f"Initializing ChromaDB at: {self.persist_dir}")

        # ✅ Ensure directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            print("✅ Persistent Chroma client initialized.")
        except Exception as e:
            print(f"⚠️ Persistent client failed ({e}), using Ephemeral fallback.")
            self.client = chromadb.EphemeralClient()
            print("⚡ Running in-memory mode — data won't persist after restart.")

        # ✅ Create or get a collection
        try:
            self.collection = self.client.get_or_create_collection("tickets")
            print("Collection 'tickets' loaded successfully.")
        except Exception as e:
            print(f"Failed to create/get collection: {e}")
            self.collection = None

    def add_chunks(self, chunks, embeddings, metadatas):
        """Add embedded chunks into the Chroma collection."""
        if not self.collection:
            print("Collection not initialized. Cannot add chunks.")
            return

        try:
            ids = [f"doc_{i}" for i in range(len(chunks))]
            self.collection.add(
                documents=chunks,
                embeddings=[emb.tolist() for emb in embeddings],
                metadatas=metadatas,
                ids=ids
            )
            print(f"Stored {len(chunks)} chunks in ChromaDB.")
        except Exception as e:
            print(f"Failed to add chunks: {e}")

    def search(self, query_embedding, top_k=4):
        """Search for top-k similar chunks using vector similarity."""
        if not self.collection:
            print("No collection found. Cannot perform search.")
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            return list(zip(docs, metas, dists))
        except Exception as e:
            print(f"Query failed: {e}")
            return []
