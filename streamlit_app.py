import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="Ticket Analysis Bot", page_icon="🤖", layout="wide")

st.title("Ticket Analysis Bot")
st.caption("Ask questions about your ticket dataset using AI-powered retrieval and summarization.")

# User input
question = st.text_input("Enter your question", placeholder="e.g., What are common issues reported by Roca?")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Analyzing tickets..."):
            response = requests.post(API_URL, data={"question": question})
            if response.status_code == 200:
                data = response.json()
                if "answer" in data:
                    st.subheader("Answer")
                    st.write(data["answer"])
                else:
                    st.error(f"Error: {data.get('error', 'Unknown error')}")
            else:
                st.error(f"API error: {response.status_code}")
    else:
        st.warning("Please enter a question before submitting.")

st.markdown("---")
st.caption("Powered by ChromaDB + Groq LLM")
