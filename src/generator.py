"""
Generator Module: Handles LLM response generation via OpenRouter
"""
import requests
import json
from config import OPENROUTER_API_KEY, MODEL_NAME, MAX_TOKENS, TEMPERATURE


class RAGGenerator:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.model = MODEL_NAME

    def _build_prompt(self, query, context_chunks):
        context = "\n\n---\n\n".join(context_chunks)
        return f"""You are a document assistant. Your ONLY job is to answer questions using the context provided.

STRICT RULES:
- Answer ONLY using information from the context below
- Do NOT use general knowledge or knowledge outside the context
- If the context does not contain the answer, respond with: "This specific information is not in the retrieved sections. Try rephrasing your question."
- Quote directly from context when possible
- Be specific and detailed
- Never improvise or infer beyond what is explicitly stated

Context:
{context}

Question: {query}

Answer (based strictly on the context above):"""

    def generate(self, query, context_chunks):
        """Non-streaming fallback."""
        prompt = self._build_prompt(query, context_chunks)
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            },
            timeout=30,
        )
        data = response.json()

        if "choices" not in data:
            error_msg = data.get("error", {}).get("message", str(data))
            raise ValueError(f"OpenRouter API error: {error_msg}")

        return data["choices"][0]["message"]["content"]

    def stream_generate(self, query, context_chunks):
        """
        Streams tokens. If the stream returns nothing,
        falls back to non-streaming automatically.
        """
        prompt = self._build_prompt(query, context_chunks)

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8501",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "stream": True,
                },
                stream=True,
                timeout=60,
            )

            yielded_anything = False

            for line in response.iter_lines():
                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                if not line.startswith("data:"):
                    continue

                payload = line[5:].strip()

                if payload == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload)
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    text = delta.get("content", "")

                    if not text:
                        text = choice.get("text", "")

                    if text:
                        yielded_anything = True
                        yield text

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

            if not yielded_anything:
                answer = self.generate(query, context_chunks)
                # Fallback: yield word-by-word for smooth streaming effect
                for word in answer.split(" "):
                    if word.strip():
                        yield word + " "

        except Exception as e:
            try:
                answer = self.generate(query, context_chunks)
                yield answer
            except Exception:
                yield f"Error contacting the model: {str(e)}"

