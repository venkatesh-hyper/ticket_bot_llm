import os, requests, textwrap

class LLMClient:
    """Groq LLM client with fallback mode and detailed error handling."""

    def __init__(self, api_key=None, model="mixtral-8x7b"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "mixtral-8x7b")

        if not self.api_key:
            print("⚠️ No GROQ_API_KEY found. Running in offline summarizer mode.")
            self.offline = True
        else:
            self.offline = False

    def generate_answer(self, context_chunks, question):
        """Generate summarized answer using Groq or offline mode."""
        if self.offline:
            joined = " ".join(context_chunks[:3])
            snippet = textwrap.shorten(joined, width=800, placeholder=" ...")
            return f"[Offline summary]\n{snippet}"

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json"
        }

        prompt = (
            "You are a support ticket analysis assistant. "
            "Use only the context provided to answer accurately.\n\n"
            f"Context:\n{context_chunks}\n\n"
            f"Question: {question}"
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 512
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            data = response.json()

            if response.status_code == 401:
                return "Invalid or expired API key. Please check your GROQ_API_KEY."
            if "choices" in data:
                return data["choices"][0]["message"]["content"]

            return f"⚠️ Unexpected response: {data}"

        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Try again later."
        except Exception as e:
            return f"LLM call failed: {e}"
