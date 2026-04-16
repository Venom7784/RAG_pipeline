from __future__ import annotations

from typing import Any
from pathlib import Path
from urllib.parse import quote
import re

import chainlit as cl
from chainlit.server import app
from fastapi import HTTPException
from fastapi.responses import FileResponse

from scr.agent import create_app_agent, get_agent_text, invoke_agent
from scr.config import PipelineConfig
from scr.services.pipeline_service import PipelineService


service = PipelineService(config=PipelineConfig())
agent = None
DEBUG_COMMANDS = {"/debug on", "/debug off", "/debug status"}
SOURCE_LINE_RE = re.compile(
    r"^\d+\.\s+file=(?P<file>.+?)\s+\|\s+page=(?P<page>\d+)\s+\|\s+score=(?P<score>\d+(?:\.\d+)?)$"
)


def _resolve_pdf_path(file_name: str) -> Path:
    safe_name = Path(file_name).name
    pdf_root = service.config.pdf_directory.resolve()
    pdf_path = (pdf_root / safe_name).resolve()

    try:
        pdf_path.relative_to(pdf_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid PDF path.") from exc

    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found.")

    return pdf_path


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p).strip()
    return str(content or "")


def _extract_agent_debug(response: dict[str, Any]) -> tuple[list[str], list[str]]:
    messages = response.get("messages") or []
    tool_calls: list[str] = []
    source_lines: list[str] = []

    for msg in messages:
        if isinstance(msg, dict):
            msg_type = str(msg.get("type", ""))
            role = str(msg.get("role", ""))
            name = str(msg.get("name", ""))
            text = _content_to_text(msg.get("content", ""))
        else:
            msg_type = str(getattr(msg, "type", ""))
            role = str(getattr(msg, "role", ""))
            name = str(getattr(msg, "name", ""))
            text = _content_to_text(getattr(msg, "content", ""))

        if msg_type == "tool" or role == "tool":
            tool_calls.append(name or "unknown_tool")
            if "Sources:" in text:
                _, _, source_section = text.partition("Sources:")
                for line in source_section.strip().splitlines():
                    cleaned = line.strip()
                    if cleaned:
                        source_lines.append(cleaned)

    return tool_calls, source_lines


def _extract_agent_sources(response: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    messages = response.get("messages") or []
    tool_calls: list[str] = []
    sources: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, dict):
            msg_type = str(msg.get("type", ""))
            role = str(msg.get("role", ""))
            name = str(msg.get("name", ""))
            text = _content_to_text(msg.get("content", ""))
        else:
            msg_type = str(getattr(msg, "type", ""))
            role = str(getattr(msg, "role", ""))
            name = str(getattr(msg, "name", ""))
            text = _content_to_text(getattr(msg, "content", ""))

        if msg_type != "tool" and role != "tool":
            continue

        tool_name = name or "unknown_tool"
        tool_calls.append(tool_name)
        if tool_name != "rag_search" or "Sources:" not in text:
            continue

        _, _, source_section = text.partition("Sources:")
        for line in source_section.strip().splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.lower() == "none":
                continue
            match = SOURCE_LINE_RE.match(cleaned)
            if not match:
                continue
            sources.append(
                {
                    "file_name": Path(match.group("file")).name,
                    "page": int(match.group("page")),
                    "score": float(match.group("score")),
                }
            )

    return tool_calls, sources


def _build_source_markdown(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "Sources: none"

    lines = ["Sources:"]
    for idx, source in enumerate(sources, start=1):
        file_name = source["file_name"]
        page = int(source["page"])
        score = float(source["score"])
        lines.append(
            f"{idx}. {file_name} (page {page}, score {score:.3f})"
        )
    return "\n".join(lines)


def _build_source_elements(sources: list[dict[str, Any]]) -> list[cl.Pdf]:
    elements: list[cl.Pdf] = []

    for idx, source in enumerate(sources, start=1):
        file_name = source["file_name"]
        page = int(source["page"])
        pdf_path = _resolve_pdf_path(file_name)
        elements.append(
            cl.Pdf(
                name=f"{idx}. {file_name} (page {page})",
                path=str(pdf_path),
                display="side",
                page=page,
            )
        )

    return elements


def _build_source_actions(sources: list[dict[str, Any]]) -> list[cl.Action]:
    actions: list[cl.Action] = []

    for idx, source in enumerate(sources, start=1):
        file_name = source["file_name"]
        page = int(source["page"])
        actions.append(
            cl.Action(
                name="open_pdf_source",
                label=f"Open Source {idx}",
                tooltip=f"Open {file_name} at page {page}",
                payload={
                    "file_name": file_name,
                    "page": page,
                    "source_index": idx,
                },
            )
        )

    return actions


@cl.action_callback("open_pdf_source")
async def open_pdf_source(action: cl.Action) -> None:
    payload = action.payload or {}
    file_name = Path(str(payload.get("file_name", ""))).name
    page = int(payload.get("page", 1))
    source_index = int(payload.get("source_index", 1))
    pdf_path = _resolve_pdf_path(file_name)

    pdf_element = cl.Pdf(
        name=f"{source_index}. {file_name} (page {page})",
        path=str(pdf_path),
        display="side",
        page=page,
    )
    await cl.ElementSidebar.set_title(f"Source {source_index}")
    await cl.ElementSidebar.set_elements([pdf_element], key=f"{file_name}:{page}")


def _is_pdf_file(file: cl.File) -> bool:
    mime = (file.mime or "").lower()
    name = (file.name or "").lower()
    path = (file.path or "").lower()
    return mime == "application/pdf" or name.endswith(".pdf") or path.endswith(".pdf")


def _get_pdf_attachments(message: cl.Message) -> list[cl.File]:
    elements = getattr(message, "elements", None) or []
    pdf_files: list[cl.File] = []

    for element in elements:
        if isinstance(element, cl.File) and _is_pdf_file(element):
            pdf_files.append(element)

    return pdf_files


async def handle_pdf_attachments(files: list[cl.File]) -> None:
    progress = cl.Message(content=f"Saving and indexing {len(files)} attached PDF(s)...")
    await progress.send()

    successes: list[str] = []
    failures: list[str] = []
    latest_vector_count: int | None = None

    for uploaded in files:
        try:
            stored_path = await cl.make_async(service.save_uploaded_pdf)(
                uploaded.path,
                uploaded.name,
            )
            summary = await cl.make_async(service.ingest_pdf)(str(stored_path))
            successes.append(
                f"`{summary['file_name']}` (pages: `{summary['pages']}`, chunks: `{summary['chunks']}`)"
            )
            latest_vector_count = summary["vector_count"]
        except Exception as exc:
            failures.append(f"`{uploaded.name}`: {exc}")

    lines = []
    if successes:
        lines.append("Added PDFs:")
        lines.extend(f"- {item}" for item in successes)
    if failures:
        lines.append("Failed PDFs:")
        lines.extend(f"- {item}" for item in failures)
    if latest_vector_count is not None:
        lines.append(f"Total indexed chunks: `{latest_vector_count}`")

    progress.content = "\n".join(lines) if lines else "No PDF attachments were processed."
    await progress.update()


@app.get("/pdf-file/{file_name}")
async def pdf_file(file_name: str):
    pdf_path = _resolve_pdf_path(file_name)
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)


@cl.on_chat_start
async def on_chat_start() -> None:
    global agent

    status_message = cl.Message(content="Preparing the RAG assistant...")
    await status_message.send()

    try:
        agent = create_app_agent(config=service.config, service=service)
    except Exception as exc:
        status_message.content = (
            "I couldn't initialize the assistant. "
            f"Please check your configuration and documents.\n\nError: {exc}"
        )
        await status_message.update()
        return

    status = service.get_status()
    cl.user_session.set("debug_mode", False)
    status_message.content = (
        "RAG assistant is ready.\n\n"
        f"- Collection: `{status['collection_name']}`\n"
        f"- Indexed chunks already stored: `{status['vector_count']}`\n"
        f"- PDF directory: `{status['pdf_directory']}`\n"
        "- Pipeline loading: `lazy` (loads on first RAG search or PDF upload)\n"
        "- Debug: `off` (use `/debug on`)\n\n"
        "Attach a PDF with the paperclip button any time to add it to the knowledge base."
    )
    await status_message.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    global agent

    attached_pdfs = _get_pdf_attachments(message)
    if attached_pdfs:
        await handle_pdf_attachments(attached_pdfs)

    query = _content_to_text(message.content).strip()
    if not query:
        if not attached_pdfs:
            await cl.Message(content="Please enter a question about your documents.").send()
        return

    lower_query = query.lower()
    if lower_query in DEBUG_COMMANDS:
        if lower_query == "/debug on":
            cl.user_session.set("debug_mode", True)
            await cl.Message(
                content="Debug mode is now ON. I will show tool calls and PDF sources."
            ).send()
            return
        if lower_query == "/debug off":
            cl.user_session.set("debug_mode", False)
            await cl.Message(content="Debug mode is now OFF.").send()
            return
        debug_enabled = bool(cl.user_session.get("debug_mode"))
        await cl.Message(
            content=f"Debug mode is currently `{'on' if debug_enabled else 'off'}`."
        ).send()
        return

    thinking = cl.Message(content="Searching your documents and drafting an answer...")
    await thinking.send()

    try:
        if agent is None:
            agent = create_app_agent(config=service.config, service=service)
        response = await cl.make_async(invoke_agent)(agent, query)
        answer_text = get_agent_text(response).strip()
        if not answer_text:
            raise ValueError("Agent returned an empty response.")
        tool_calls, sources = _extract_agent_sources(response)
        if "rag_search" in tool_calls:
            if sources:
                source_markdown = _build_source_markdown(sources)
                if "Sources:" not in answer_text:
                    answer_text = f"{answer_text}\n\n{source_markdown}"
        if bool(cl.user_session.get("debug_mode")):
            _, source_lines = _extract_agent_debug(response)
            tool_line = (
                f"Tool called: `{', '.join(tool_calls)}`"
                if tool_calls
                else "Tool called: `none`"
            )
            sources_block = (
                "\n".join(source_lines) if source_lines else "No PDF sources captured."
            )
            answer_text = (
                f"{answer_text}\n\n---\n"
                f"Debug\n{tool_line}\nSources\n{sources_block}"
            )
        thinking.content = answer_text
        if sources:
            thinking.actions = _build_source_actions(sources)
            await cl.ElementSidebar.set_title("Sources")
            await cl.ElementSidebar.set_elements(_build_source_elements(sources), key=thinking.id)
        await thinking.update()
        return
    except Exception as exc:
        thinking.content = f"I ran into an error while answering: {exc}"
        await thinking.update()
