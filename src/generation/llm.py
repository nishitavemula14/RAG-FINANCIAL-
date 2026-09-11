import os
import re
from typing import List, Any, Optional
from src.generation.prompt import SYSTEM_PROMPT, build_rag_prompt

def extractive_fallback(question: str, found_chunks: List[Any]) -> str:
    """
    Local fallback: finds and returns the most relevant sentence from retrieved chunks
    when LLM generation is unavailable or fails.
    """
    terms = set(re.findall(r"\w+", question.lower()))
    all_text = " ".join(c.text for c in found_chunks)
    sentences = re.split(r"(?<=[.!?])\s+", all_text)
    best = max(
        sentences,
        key=lambda sentence: len(terms & set(re.findall(r"\w+", sentence.lower()))),
        default=""
    )
    return best.strip() or "No relevant information was found."

class LLMGenerator:
    """Manages LLM completions (Groq, OpenAI) with automatic extractive fallback."""
    def __init__(
        self,
        provider: str = "groq",
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.1,
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature

    def generate(self, question: str, context: str, found_chunks: List[Any]) -> str:
        """Attempts API generation via Groq or OpenAI, falling back to extractive matching."""
        prompt = build_rag_prompt(question, context)

        # 1. Try Groq
        if self.provider == "groq" and os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                client = Groq()
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"Warning: Groq completion failed: {e}")

        # 2. Try OpenAI
        if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"Warning: OpenAI completion failed: {e}")

        # 3. Fallback: Extractive sentence selection
        return extractive_fallback(question, found_chunks)
