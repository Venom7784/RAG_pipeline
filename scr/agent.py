from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from .config import PipelineConfig
from .services.pipeline_service import PipelineService
from .tools import rag_search, set_rag_search_service

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for a PDF RAG application. "
    "When a question depends on the uploaded PDFs, call the `rag_search` tool first "
    "and ground your final answer in its results. "
    "If the user asks a general question not related to the PDFs, answer directly. "
    "Answer clearly and briefly. If you do not know the answer, say so."
)


def _create_llm(
    model_name: str,
    temperature: float,
    max_tokens: int,
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


def create_app_agent(
    config: PipelineConfig | None = None,
    service: PipelineService | None = None,
    tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    debug: bool = False,
):
    app_config = config or PipelineConfig()
    app_service = service or PipelineService(config=app_config)
    set_rag_search_service(service=app_service, config=app_config)
    resolved_tools = (
        list(tools)
        if tools is not None
        else [rag_search]
    )
    llm = _create_llm(
        model_name=app_config.groq_model_name,
        temperature=app_config.temperature,
        max_tokens=app_config.max_tokens,
    )
    return create_agent(
        model=llm,
        tools=resolved_tools,
        system_prompt=system_prompt,
        debug=debug,
        name="rag_pipeline_agent",
    )


def invoke_agent(agent: Any, user_message: str) -> dict[str, Any]:
    return agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        }
    )


def get_agent_text(response: dict[str, Any]) -> str:
    messages = response.get("messages")
    if not messages:
        return ""

    last_message = messages[-1]
    content = getattr(last_message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "\n".join(part for part in text_parts if part).strip()

    return str(content)
