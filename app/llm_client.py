import os, requests, textwrap

class LLMClient:
    """Groq or fallback LLM client."""

    def __init__(self, api_key=None, model="mixtral-8x7b"):
        self.api_key = api_key
        self.model = model
        if not api_key:
            print(" No GROQ_API_KEY found. Using offline summarizer mode.")

    def generate_answer(self, context_chunks, question):
        """Generate a final summarized answer using the LLM."""
        # Offline fallback (if no key)
        if not self.api_key:
            joined = " ".join(context_chunks[:3])
            snippet = textwrap.shorten(joined, width=800, placeholder=" ...")
            return f"[Offline summary] Based on retrieved text:\n\n{snippet}"

        # Online Groq API call
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        prompt = (
            "You are a support ticket analysis assistant. "
            "Answer the user's question using only the provided context. "
            "If the answer is not present, say 'Not found in the provided data.'\n\n"
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
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            return f"Unexpected response: {data}"
        except Exception as e:
            return f" LLM call failed: {e}"
