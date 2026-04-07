from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import chainlit as cl
from chainlit.server import app
from fastapi import HTTPException
from fastapi.responses import FileResponse

from scr.config import PipelineConfig
from scr.services.pipeline_service import PipelineService


service = PipelineService(config=PipelineConfig())
UPLOAD_COMMANDS = {"/upload", "upload pdf", "add pdf"}


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


def _build_pdf_file_url(metadata: dict) -> str | None:
    source_file = metadata.get("source_file")
    if not source_file:
        return None

    page = int(metadata.get("page", 0)) + 1
    return f"/pdf-file/{quote(Path(source_file).name)}#page={page}"


@app.get("/pdf-file/{file_name}")
async def pdf_file(file_name: str):
    pdf_path = _resolve_pdf_path(file_name)
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)


@cl.on_chat_start
async def on_chat_start() -> None:
    status_message = cl.Message(content="Preparing the RAG pipeline...")
    await status_message.send()

    try:
        await cl.make_async(service.ensure_pipeline)()
    except Exception as exc:
        status_message.content = (
            "I couldn't initialize the pipeline. "
            f"Please check your configuration and documents.\n\nError: {exc}"
        )
        await status_message.update()
        return

    status = service.get_status()
    status_message.content = (
        "RAG assistant is ready.\n\n"
        f"- Collection: `{status['collection_name']}`\n"
        f"- Indexed chunks: `{status['vector_count']}`\n"
        f"- PDF directory: `{status['pdf_directory']}`\n\n"
        "Type `/upload` any time to add a PDF."
    )
    await status_message.update()


async def handle_pdf_upload() -> None:
    files = await cl.AskFileMessage(
        content="Upload a PDF to add it to the knowledge base.",
        accept=["application/pdf"],
        max_files=1,
        max_size_mb=25,
        timeout=180,
    ).send()

    if not files:
        await cl.Message(content="Upload cancelled or timed out.").send()
        return

    uploaded = files[0]
    progress = cl.Message(content=f"Saving and indexing `{uploaded.name}`...")
    await progress.send()

    try:
        stored_path = await cl.make_async(service.save_uploaded_pdf)(
            uploaded.path,
            uploaded.name,
        )
        summary = await cl.make_async(service.ingest_pdf)(str(stored_path))
    except Exception as exc:
        progress.content = f"I couldn't add that PDF: {exc}"
        await progress.update()
        return

    progress.content = (
        f"Added `{summary['file_name']}` successfully.\n\n"
        f"- Pages loaded: `{summary['pages']}`\n"
        f"- Chunks added: `{summary['chunks']}`\n"
        f"- Total indexed chunks: `{summary['vector_count']}`"
    )
    await progress.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = message.content.strip()
    if not query:
        await cl.Message(content="Please enter a question about your documents.").send()
        return

    if query.lower() in UPLOAD_COMMANDS:
        await handle_pdf_upload()
        return

    thinking = cl.Message(content="Searching your documents and drafting an answer...")
    await thinking.send()

    try:
        result = await cl.make_async(service.query)(query)
    except Exception as exc:
        thinking.content = f"I ran into an error while answering: {exc}"
        await thinking.update()
        return

    elements = []
    for index, doc in enumerate(result["results"], start=1):
        metadata = doc["metadata"]
        pdf_url = _build_pdf_file_url(metadata)
        page_number = int(metadata.get("page", 0)) + 1
        source_name = (
            metadata.get("source_file")
            or metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or f"Source {index}"
        )
        source_lines = [
            f"Similarity score: {doc['similarity_score']:.3f}",
            f"Page: {page_number}",
            "",
            doc["content"],
            "",
            "Metadata:",
            str(metadata),
        ]
        if pdf_url:
            source_lines.extend(["", f"Open PDF: {pdf_url}"])
        elements.append(
            cl.Text(
                name=f"source_{index}",
                content="\n".join(source_lines),
                display="side",
            )
        )
        if metadata.get("source_file") and pdf_url:
            pdf_path = _resolve_pdf_path(metadata["source_file"])
            elements.append(
                cl.Pdf(
                    name=f"{source_name} - page {page_number}",
                    path=str(pdf_path),
                    page=page_number,
                    display="side",
                )
            )

    source_summary = ""
    if result["results"]:
        bullets = []
        for index, doc in enumerate(result["results"], start=1):
            metadata = doc["metadata"]
            source_name = (
                metadata.get("source_file")
                or metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("filename")
                or f"Source {index}"
            )
            page_number = int(metadata.get("page", 0)) + 1
            pdf_url = _build_pdf_file_url(metadata)
            if pdf_url:
                bullets.append(
                    f"{index}. [{source_name} - page {page_number}]({pdf_url}) "
                    f"({doc['similarity_score']:.3f})"
                )
            else:
                bullets.append(
                    f"{index}. `{source_name}` page {page_number} "
                    f"({doc['similarity_score']:.3f})"
                )
        source_summary = "\n\nSources\n" + "\n".join(bullets)

    thinking.content = f"{result['answer']}{source_summary}"
    thinking.elements = elements
    await thinking.update()
