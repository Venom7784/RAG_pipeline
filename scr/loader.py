from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def process_pdf_file(file_path: str):
    """Load a single PDF file using PyPDFLoader."""
    pdf_file = Path(file_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {file_path}")

    loader = PyPDFLoader(str(pdf_file))
    document = loader.load()
    for doc in document:
        doc.metadata["source_file"] = pdf_file.name
        doc.metadata["file_type"] = "pdf"
    print(f"Loaded {len(document)} pages from {pdf_file.name}")
    return document


def process_pdfs_in_directory(directory_path: str, recursive: bool = True):
    """Load all PDFs from a directory (optionally recursively) using PyPDFLoader."""
    pdf_dir = Path(directory_path)

    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created PDF directory at {pdf_dir}")
        return []

    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory path: {directory_path}")

    documents = []

    pdf_files = pdf_dir.rglob("*.pdf") if recursive else pdf_dir.glob("*.pdf")
    for pdf_file in sorted(pdf_files):
        documents.extend(process_pdf_file(str(pdf_file)))

    return documents
