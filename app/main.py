from app.ingestion import PDFIngestor
from app.embedder import Embedder
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.llm_client import LLMClient
import yaml, os
from dotenv import load_dotenv

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    load_dotenv()
    config = load_config()

    print("🚀 Ticket Analysis Bot — Starting interactive pipeline")

    store = VectorStore(config["chroma_dir"])
    retriever = Retriever(store, config["model_name"])
    llm = LLMClient(os.getenv("GROQ_API_KEY"), os.getenv("LLM_MODEL"))

    while True:
        query = input("\nAsk a question (or 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        contexts = retriever.get_relevant_chunks(query, top_k=config["top_k"])
        answer = llm.generate_answer(contexts, query)
        print("\n Answer:\n", answer)


if __name__ == "__main__":
    main()
