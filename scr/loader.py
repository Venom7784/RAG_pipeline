from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def process_pdfs_in_directory(directory_path: str):
    """Load all PDFs from a directory using PyPDFLoader."""
    pdf_dir = Path(directory_path)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory path: {directory_path}")

    documents = []

    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_file))
        document = loader.load()
        for doc in document:
            doc.metadata["source_file"] = pdf_file.name
            doc.metadata["file_type"] = "pdf"
        documents.extend(document)
        print(f"Loaded {len(document)} pages from {pdf_file.name}")

    return documents
