import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class PipelineConfig:
    pdf_directory: Path = Path(os.getenv("PDF_DIRECTORY", BASE_DIR / "data" / "pdf"))
    persist_directory: Path = Path(
        os.getenv("PERSIST_DIRECTORY", BASE_DIR / "data" / "vector_store")
    )
    collection_name: str = os.getenv("COLLECTION_NAME", "pdf_documents")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
    retrieval_results: int = int(os.getenv("RETRIEVAL_RESULTS", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    def __post_init__(self):
        self.pdf_directory = Path(self.pdf_directory)
        self.persist_directory = Path(self.persist_directory)
