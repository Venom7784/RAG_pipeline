import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from scr.config import PipelineConfig
from scr.loader import process_pdf_file
from scr.pipeline import initialize_pipeline, rag_simple
from scr.splitter import split_documents


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

    def query(self, query: str, n_results: int | None = None):
        pipeline = self.ensure_pipeline()
        result_count = n_results or self.config.retrieval_results
        rag_result = rag_simple(
            query=query,
            llm=pipeline["llm"],
            retriever=pipeline["rag_retriever"],
            config=self.config,
            n_results=result_count,
        )
        return {
            "query": query,
            "answer": rag_result["answer"],
            "results": rag_result["results"],
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
