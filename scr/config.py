import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DESKTOP_MODE_ENV = "RAG_DESKTOP_MODE"
WINDOWS_APP_DIR = "RAGPipeline"


def _desktop_data_root() -> Path:
    appdata_root = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
    if appdata_root:
        return Path(appdata_root) / WINDOWS_APP_DIR / "data"
    return BASE_DIR / "data-desktop"


def resolve_data_root() -> Path:
    explicit_root = os.getenv("DATA_ROOT")
    if explicit_root:
        return Path(explicit_root)

    if os.getenv(DESKTOP_MODE_ENV) == "1":
        return _desktop_data_root()

    return BASE_DIR / "data"


def _default_pdf_directory() -> Path:
    return Path(os.getenv("PDF_DIRECTORY", resolve_data_root() / "pdf"))


def _default_persist_directory() -> Path:
    return Path(os.getenv("PERSIST_DIRECTORY", resolve_data_root() / "vector_store"))


@dataclass(slots=True)
class PipelineConfig:
    pdf_directory: Path = field(default_factory=_default_pdf_directory)
    persist_directory: Path = field(default_factory=_default_persist_directory)
    collection_name: str = os.getenv("COLLECTION_NAME", "pdf_documents")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
    retrieval_results: int = int(os.getenv("RETRIEVAL_RESULTS", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))

    def __post_init__(self):
        self.pdf_directory = Path(self.pdf_directory)
        self.persist_directory = Path(self.persist_directory)
        self.pdf_directory.mkdir(parents=True, exist_ok=True)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
