import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from scr.config import PipelineConfig
from scr.loader import process_pdf_file
from scr.pipeline import initialize_pipeline
from scr.splitter import split_documents
from scr.vector_store import VectorStore


class PipelineService:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.pipeline = None
        self.last_built_at = None
        self.last_error = None
        self._lock = Lock()

    def build(self):
        with self._lock:
            try:
                self.pipeline = initialize_pipeline(self.config)
                self.last_built_at = datetime.now(timezone.utc)
                self.last_error = None
                return self.pipeline
            except Exception as exc:
                self.last_error = str(exc)
                raise

    def ensure_pipeline(self):
        if self.pipeline is None:
            return self.build()
        return self.pipeline

    def get_status(self):
        pipeline = self.pipeline
        vector_count = 0

        if pipeline is not None:
            vector_count = pipeline["vector_store"].count()
        else:
            vector_count = self.get_persisted_vector_count()

        return {
            "ready": pipeline is not None,
            "collection_name": self.config.collection_name,
            "pdf_directory": str(self.config.pdf_directory),
            "persist_directory": str(self.config.persist_directory),
            "retrieval_results": self.config.retrieval_results,
            "similarity_threshold": self.config.similarity_threshold,
            "vector_count": vector_count,
            "last_built_at": (
                self.last_built_at.isoformat() if self.last_built_at is not None else None
            ),
            "last_error": self.last_error,
        }

    def get_persisted_vector_count(self) -> int:
        vector_store = VectorStore(
            collection=self.config.collection_name,
            persist_directory=str(self.config.persist_directory),
        )
        return vector_store.count()

    def query(self, query: str, n_results: int | None = None):
        pipeline = self.ensure_pipeline()
        result_count = n_results or self.config.retrieval_results
        results = pipeline["rag_retriever"].retriever(
            query=query,
            n_results=result_count,
            threshold=self.config.similarity_threshold,
        )
        context = self._build_context(results)
        return {
            "query": query,
            "retrieval_query": query.strip(),
            "context": context,
            "results": results,
        }

    def retrieve(
        self,
        query: str,
        n_results: int | None = None,
        threshold: float | None = None,
    ):
        pipeline = self.ensure_pipeline()
        return pipeline["rag_retriever"].retriever(
            query=query,
            n_results=n_results or self.config.retrieval_results,
            threshold=threshold if threshold is not None else self.config.similarity_threshold,
        )

    def save_uploaded_pdf(self, uploaded_path: str, original_name: str) -> Path:
        source = Path(uploaded_path)
        destination_dir = self.config.pdf_directory
        destination_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(original_name).name
        destination = destination_dir / safe_name
        counter = 1
        while destination.exists():
            destination = destination_dir / f"{Path(safe_name).stem}_{counter}.pdf"
            counter += 1

        shutil.copy2(source, destination)
        return destination

    def ingest_pdf(self, pdf_path: str):
        documents = process_pdf_file(pdf_path)
        split_docs = split_documents(
            documents=documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        if not split_docs:
            raise ValueError("No text chunks were generated from the uploaded PDF.")

        if self.pipeline is None:
            pipeline = self.build()
            return {
                "file_name": Path(pdf_path).name,
                "pages": len(documents),
                "chunks": len(split_docs),
                "vector_count": pipeline["vector_store"].count(),
            }

        pipeline = self.pipeline
        texts = [doc.page_content for doc in split_docs]
        embeddings = pipeline["embedding_manager"](texts)
        pipeline["vector_store"].add_documents(split_docs, embeddings)
        self.last_built_at = datetime.now(timezone.utc)
        self.last_error = None

        return {
            "file_name": Path(pdf_path).name,
            "pages": len(documents),
            "chunks": len(split_docs),
            "vector_count": pipeline["vector_store"].count(),
        }

    def _build_context(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return ""

        context_parts: list[str] = []
        for idx, doc in enumerate(results, start=1):
            metadata = doc.get("metadata", {}) or {}
            source_name = (
                metadata.get("source_file")
                or metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("filename")
                or f"Source {idx}"
            )
            page_number = int(metadata.get("page", 0)) + 1
            context_parts.append(
                f"[Source {idx}: {source_name}, page {page_number}]\n{doc.get('content', '').strip()}"
            )

        return "\n\n".join(part for part in context_parts if part.strip())
