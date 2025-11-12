#!/bin/bash
echo "Starting FastAPI backend..."
uvicorn app.api:app --host 127.0.0.1 --port 8000 &

# wait 5 seconds for backend to boot up
sleep 5

echo "Starting Streamlit frontend..."
streamlit run streamlit_app.py --server.port 8501
