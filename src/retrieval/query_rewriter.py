import os
import re

class QueryRewriter:
    """
    Cleans, fixes grammatical errors, corrects typos, and normalizes queries
    before passing them to the retrieval pipeline.
    """
    def __init__(
        self,
        provider: str = "groq",
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.0,
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature

    def rewrite(self, raw_query: str) -> str:
        """
        Rewrites the query using an LLM if available, fixing grammar and typos.
        Falls back to local basic normalization if offline or if the API call fails.
        """
        query = raw_query.strip()
        if not query:
            return ""

        # Basic local cleanup (remove duplicate spaces, strip erratic punctuation)
        clean_local = re.sub(r"\s+", " ", query).strip()

        # Attempt LLM rewriting if keys are available
        prompt = (
            "You are a search query optimizer. Correct any grammatical mistakes, "
            "spelling typos, slang, or awkward phrasing in the user's query so that it becomes "
            "a clear, grammatically correct search query for a financial and document retrieval system. "
            "Do NOT answer the question. Output ONLY the corrected query text, with no quotes or explanation.\n\n"
            f"User Query: {clean_local}"
        )

        # 1. Try Groq
        if self.provider == "groq" and os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                client = Groq()
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                rewritten = resp.choices[0].message.content.strip().strip('"').strip("'")
                if rewritten:
                    return rewritten
            except Exception as e:
                pass

        # 2. Try OpenAI
        if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                rewritten = resp.choices[0].message.content.strip().strip('"').strip("'")
                if rewritten:
                    return rewritten
            except Exception as e:
                pass

        # Fallback: return cleaned local query
        return clean_local

