from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.llm_client import LLMClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(title="Ticket Analysis Bot API", version="1.1")

# Allow CORS for Streamlit or external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧩 Initialize components
print("Initializing Ticket Analysis Bot components...")
store = VectorStore("chroma_storage")
retriever = Retriever(store)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM client with fallbacks
GROQ_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
llm = LLMClient(api_key=GROQ_KEY, model=MODEL_NAME)
print(f"Using model: {MODEL_NAME}\n")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Ticket Analysis Bot API is running", "model": MODEL_NAME}


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Main Q&A endpoint — retrieves context and queries Groq LLM."""
    try:
        print(f" Received question: {question}")

        # 1️⃣ Retrieve relevant context from Chroma
        contexts = retriever.get_relevant_chunks(question)
        context_count = len(contexts)
        print(f"Retrieved {context_count} context chunks from ChromaDB")

        if not contexts:
            return {
                "question": question,
                "answer": "No context retrieved from ChromaDB. Try re-ingesting or broadening your query.",
                "context_count": 0,
                "success": False
            }

        #  Generate contextual answer
        answer = llm.generate_answer(contexts, question)

        # 3 Return well-structured response
        return {
            "question": question,
            "answer": answer,
            "context_count": context_count,
            "success": True
        }

    except Exception as e:
        print(f"Error during /ask: {e}")
        return {
            "error": str(e),
            "question": question,
            "success": False
        }
