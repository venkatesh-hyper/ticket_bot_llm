# Ticket Analysis Bot Tech Support Team  
> AI-powered ticket analytics and summarization using LLMs, ChromaDB, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-success?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)
![Groq](https://img.shields.io/badge/LLM-Groq-lightgrey?logo=openai)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## Overview
**Ticket Analysis Bot** is an end-to-end **Retrieval-Augmented Generation (RAG)** system designed to process and analyze large-scale support ticket PDFs.  
It extracts, embeds, and indexes textual data from multi-megabyte PDFs, stores embeddings in **ChromaDB**, and uses **Groq LLMs** to generate summarized, human-readable insights.

This system features:
- **FastAPI backend** for API-based queries.
- **Streamlit dashboard** for interactive exploration.
- **Modular architecture** for easy extension and scaling.

---

## Key Features
✅ Handles **70MB+ PDF documents** efficiently.  
✅ Embeds and indexes text using **SentenceTransformers**.  
✅ **Vector search** powered by ChromaDB.  
✅ Summarizes results using **Groq LLM**.  
✅ Dual access:
  - REST API (`FastAPI`)
  - Web UI (`Streamlit`)  
✅ **Docker-ready** for quick deployment.

---

## System Architecture
``` bash
flowchart TD
    A[User] -->|Question| B[Streamlit UI]
    B --> C[FastAPI Backend]
    C --> D[Retriever (ChromaDB)]
    D --> E[Vector Store]
    C --> F[LLM Client (Groq API)]
    F --> G[Answer Generator]
    G --> B
```

---

## ⚙️ Tech Stack
| Category | Technology |
|-----------|-------------|
| Language | Python 3.11 |
| Backend | FastAPI |
| Frontend | Streamlit |
| Vector Database | ChromaDB |
| Embeddings | SentenceTransformers |
| LLM | Groq (`meta-llama/llama-4-scout-17b-16e-instruct`) |
| Deployment | Docker |
| Version Control | Git + GitHub |

---

## Project Workflow
1. **PDF Ingestion:** Extracts text using PyPDF2.  
2. **Chunking & Embedding:** Generates embeddings with SentenceTransformer.  
3. **Vector Storage:** Persists embeddings in ChromaDB.  
4. **Retrieval:** Finds top-N relevant chunks for a user query.  
5. **LLM Summarization:** Generates concise, context-based answers.  
6. **Response Delivery:** Displays insights through API or Streamlit UI.  

---

## Installation & Setup

### Clone repository
```bash
git clone https://github.com/venkatesh-hyper/ticket_bot_llm.git
cd ticket_bot_llm
```

### Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure environment
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL= meta-llama/llama-4-scout-17b-16e-instruct

### Initialize config
Modify `config.yaml` as needed:
```yaml
model_name: "all-MiniLM-L6-v2"
chroma_dir: "chroma_storage"
pdf_chunk_size: 800
pdf_chunk_overlap: 100
top_k: 4
```

---

## 🚀 Running the Application

### Option 1: Run both (API + UI)
```bash
chmod +x run_all.sh
./run_all.sh
```

**Access:**
- API → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Streamlit → [http://localhost:8501](http://localhost:8501)

### Option 2: Run manually
```bash
# Terminal 1
uvicorn app.api:app --reload

# Terminal 2
streamlit run streamlit_app.py
```

---

## Example Queries
| Query | Expected Response |
|-------|--------------------|
| *“What are the most common issues from brand customers?”* | Summarized ticket patterns for Roca. |
| *“Summarize open tickets from November 2024.”* | Short summary by date filter. |
| *“Which companies reported frequent payment issues?”* | Company-wise problem breakdown. |

---

## Project Structure

```
ticket-bot/
│
├── app/
│   ├── api.py              # FastAPI endpoints
│   ├── ingestion.py        # PDF ingestion
│   ├── embedder.py         # Chunking + embeddings
│   ├── vector_store.py     # ChromaDB integration
│   ├── retriever.py        # Vector retrieval logic
│   ├── llm_client.py       # Groq LLM interface
│   ├── cache.py            # Caching logic
│   ├── logger.py           # Logger setup
│   └── main.py             # Core pipeline entry
│
├── streamlit_app.py        # Streamlit interface
├── run_all.sh              # Combined launcher script
├── config.yaml             # Configurations
├── requirements.txt        # Dependencies
├── .gitignore              # Ignored files
└── .env.template           # Example env file
```

---

## Docker Deployment
```bash
docker build -t ticket-bot .
docker run -p 8000:8000 -p 8501:8501 ticket-bot
```
Access:
- Streamlit → [http://localhost:8501](http://localhost:8501)
- FastAPI → [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Future Improvements
- [ ] Add LangChain for multi-step reasoning  
- [ ] Integrate Redis-based caching  
- [ ] Add filters for company/date  
- [ ] Build admin analytics dashboard  
- [ ] Deploy to Render / AWS ECS  

---

## 👨‍💻 Author
**Venkatesh P**  
Machine Learning Engineer | AI Infrastructure | Data Science  
[LinkedIn](https://www.linkedin.com/in/venkatesh-hyper)  
[GitHub](https://github.com/venkatesh-hyper)  

- [ ] Special Thanks to my friend koushick for data support  

> *“Built with Python, engineered for insight.”*
