# qa_engine.py
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt4(query: str, context: str, temperature: float = 0.2) -> str:
    system_prompt = (
        "You are a helpful, precise exam assistant. "
        "Use ONLY the provided context when possible; if unsure, say so. "
        "Cite which source doc (by name) you used when useful."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4o / gpt-4-turbo (pick per budget)
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
