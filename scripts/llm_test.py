import os, requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")
model = os.getenv("LLM_MODEL")

print("Model:", model)
print("Key loaded:", key[:10], "...")

res = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json={
        "model": model,
        "messages": [{"role": "user", "content": "Hello from Groq!"}]
    }
)
print(res.status_code, res.text)
