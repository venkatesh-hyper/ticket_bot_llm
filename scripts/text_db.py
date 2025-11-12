import sys
from sentence_transformers import SentenceTransformer
from app.vector_store import VectorStore

def test_chroma_db(persist_dir="chroma_storage", test_query="requirement gathering completed"):
    print("🔍 Checking ChromaDB contents...\n")

    # Initialize the store and embedding model
    store = VectorStore(persist_dir)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    try:
        # Count total stored chunks
        total_docs = store.collection.count()
        print(f"📦 Total documents in ChromaDB: {total_docs}")

        if total_docs == 0:
            print("❌ No data found. You may need to re-run ingestion.")
            return

        # Show a few sample documents
        print("\n🧾 Sample stored chunks:")
        sample_docs = store.collection.get(limit=3)
        for i, doc in enumerate(sample_docs["documents"], start=1):
            snippet = (doc[:180] + "…") if len(doc) > 180 else doc
            print(f"  {i}. {snippet}")

        # Test a semantic search query
        print(f"\n🧠 Running test query: '{test_query}'")
        query_embedding = embedder.encode(test_query)
        results = store.search(query_embedding, top_k=3)

        if not results:
            print("⚠️ No matching chunks found. Check model consistency or ingestion quality.")
        else:
            print("\n✅ Top retrieved chunks:")
            for i, (doc, meta, dist) in enumerate(results, start=1):
                snippet = (doc[:200] + "…") if len(doc) > 200 else doc
                print(f"  🔹 Rank {i} | Page {meta.get('page', '?')} | Dist: {dist:.4f}")
                print(f"     {snippet}\n")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    persist_dir = sys.argv[1] if len(sys.argv) > 1 else "chroma_storage"
    test_query = sys.argv[2] if len(sys.argv) > 2 else "requirement gathering completed"
    test_chroma_db(persist_dir, test_query)
