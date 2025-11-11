from fastapi import FastAPI, UploadFile, Form
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.llm_client import LLMClient
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Ticket Analysis Bot API", version="1.0")

# Initialize once (singleton style)
store = VectorStore("chroma_storage")
retriever = Retriever(store)
llm = LLMClient(os.getenv("GROQ_API_KEY"), os.getenv("LLM_MODEL", "mixtral-8x7b"))


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Answer a question based on indexed PDF data."""
    try:
        contexts = retriever.get_relevant_chunks(question)
        answer = llm.generate_answer(contexts, question)
        return {"question": question, "answer": answer, "context_count": len(contexts)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def root():
    return {"message": "Ticket Analysis Bot API is running 🚀"}
