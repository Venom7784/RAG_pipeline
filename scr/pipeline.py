from .config import PipelineConfig
from .embeddings import EmbeddingManager
from .llm import create_llm
from .loader import process_pdfs_in_directory
from .retriever import RAGRetriever
from .splitter import split_documents
from .vector_store import VectorStore


def rag_simple(
    query,
    llm,
    retriever,
    config: PipelineConfig,
    n_results: int | None = None,
    threshold: float | None = None,
):
    result_count = n_results if n_results is not None else config.retrieval_results
    similarity_threshold = (
        threshold if threshold is not None else config.similarity_threshold
    )
    results = retriever.retriever(
        query=query,
        n_results=result_count,
        threshold=similarity_threshold,
    )
    context = "\n\n".join(doc["content"] for doc in results) if results else ""
    if not context:
        return {
            "answer": "No context found to answer the question",
            "results": results,
        }
    response = llm.invoke(
        f"Answer the Question using this context question={query} context={context}"
    )
    return {
        "answer": response.content,
        "results": results,
    }


def initialize_pipeline(config: PipelineConfig):
    documents = []
    split_docs = []

    embedding_manager = EmbeddingManager(model_name=config.embedding_model_name)
    embedding_dimension = embedding_manager.model.get_sentence_embedding_dimension()
    vector_store = VectorStore(
        collection=config.collection_name,
        persist_directory=str(config.persist_directory),
        embedding_dimension=embedding_dimension,
        embedding_model_name=config.embedding_model_name,
    )

    if vector_store.count() == 0:
        documents = process_pdfs_in_directory(str(config.pdf_directory))
        if documents:
            split_docs = split_documents(
                documents=documents,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            text = [doc.page_content for doc in split_docs]
            embeddings = embedding_manager(text)
            vector_store.add_documents(split_docs, embeddings)
        else:
            embeddings = []
            print(
                f"No PDFs found in '{config.pdf_directory}'. "
                "The app will start with an empty knowledge base until files are uploaded."
            )
    else:
        embeddings = []
        print(
            f"Using existing vector store collection '{config.collection_name}' "
            f"with {vector_store.count()} documents."
        )

    rag_retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
    )
    llm = create_llm(
        model_name=config.groq_model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    return {
        "documents": documents,
        "split_docs": split_docs,
        "embedding_manager": embedding_manager,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "rag_retriever": rag_retriever,
        "llm": llm,
    }


def build_pipeline(config: PipelineConfig):
    documents = process_pdfs_in_directory(str(config.pdf_directory))
    split_docs = []
    if documents:
        split_docs = split_documents(
            documents=documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    embedding_manager = EmbeddingManager(model_name=config.embedding_model_name)
    embeddings = []
    if split_docs:
        text = [doc.page_content for doc in split_docs]
        embeddings = embedding_manager(text)
    embedding_dimension = embedding_manager.model.get_sentence_embedding_dimension()

    vector_store = VectorStore(
        collection=config.collection_name,
        persist_directory=str(config.persist_directory),
        embedding_dimension=embedding_dimension,
        embedding_model_name=config.embedding_model_name,
    )
    if split_docs:
        vector_store.add_documents(split_docs, embeddings)

    rag_retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
    )
    llm = create_llm(
        model_name=config.groq_model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    return {
        "documents": documents,
        "split_docs": split_docs,
        "embedding_manager": embedding_manager,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "rag_retriever": rag_retriever,
        "llm": llm,
    }
