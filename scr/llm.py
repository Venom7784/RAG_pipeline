import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


def create_llm(
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    max_tokens: int = 1024,
):
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY was not found in the environment.")

    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
