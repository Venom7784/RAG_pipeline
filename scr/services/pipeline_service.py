from datetime import datetime, timezone
from threading import Lock

from scr.config import PipelineConfig
from scr.pipeline import initialize_pipeline, rag_simple


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
        answer = rag_simple(
            query=query,
            llm=pipeline["llm"],
            retriever=pipeline["rag_retriever"],
            n_results=result_count,
        )
        documents = pipeline["rag_retriever"].retriever(
            query=query,
            n_results=result_count,
            threshold=self.config.similarity_threshold,
        )
        return {
            "query": query,
            "answer": answer,
            "results": documents,
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
