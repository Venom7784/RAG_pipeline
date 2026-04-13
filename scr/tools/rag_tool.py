from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from scr.config import PipelineConfig
from scr.services.pipeline_service import PipelineService


class RAGSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="User question to answer from the indexed PDF knowledge base.",
        min_length=1,
    )


_pipeline_service: PipelineService = PipelineService(config=PipelineConfig())


def set_rag_search_service(
    service: PipelineService | None = None,
    config: PipelineConfig | None = None,
) -> None:
    global _pipeline_service
    _pipeline_service = service or PipelineService(config=config or PipelineConfig())


@tool("rag_search", args_schema=RAGSearchInput)
def rag_search(query: str) -> str:
    """Search indexed PDFs and answer questions grounded in retrieved context."""

    cleaned_query = query.strip()
    print(f"[RAG_TOOL] rag_search called | query={cleaned_query[:120]!r}")
    if not cleaned_query:
        return "RAG search error: query is empty."

    try:
        result = _pipeline_service.query(cleaned_query)
    except Exception as exc:  # pragma: no cover - runtime-dependent
        return f"RAG search failed: {exc}"

    answer = str(result.get("answer", "")).strip() or "No answer generated."
    docs = result.get("results") or []
    if not docs:
        return f"{answer}\n\nSources: none"

    source_lines = []
    for idx, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata", {}) or {}
        source_name = (
            metadata.get("source_file")
            or metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or f"Source {idx}"
        )
        page_number = int(metadata.get("page", 0)) + 1
        score = float(doc.get("similarity_score", 0.0))
        source_lines.append(f"{idx}. {source_name} (page {page_number}, score {score:.3f})")

    print(f"[RAG_TOOL] rag_search completed | sources={len(source_lines)}")
    for line in source_lines:
        print(f"[RAG_TOOL] source: {line}")

    return f"{answer}\n\nSources:\n" + "\n".join(source_lines)
