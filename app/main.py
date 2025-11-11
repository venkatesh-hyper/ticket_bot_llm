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
    print("Ticket Analysis Bot — Initializing system...")

    pdf_ingestor = PDFIngestor()
    embedder = Embedder(config["model_name"])
    vector_db = VectorStore(config["chroma_dir"])
    retriever = Retriever(vector_db)
    llm = LLMClient(os.getenv("GROQ_API_KEY"), os.getenv("LLM_MODEL"))

    print(" System initialized successfully!")

if __name__ == "__main__":
    main()
