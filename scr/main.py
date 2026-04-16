import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scr.config import PipelineConfig
else:
    from .config import PipelineConfig


def main():
    if __package__ is None or __package__ == "":
        from scr.pipeline import initialize_pipeline
    else:
        from .pipeline import initialize_pipeline

    config = PipelineConfig()
    print("Initializing PDF RAG pipeline...")
    pipeline = initialize_pipeline(config)
    print("Pipeline ready.")
    print("Type your question and press Enter. Type 'exit' to quit.")

    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            print("Please enter a question.")
            continue

        if query.lower() in {"exit", "quit"}:
            print("Exiting CLI.")
            break

        results = pipeline["rag_retriever"].retriever(
            query=query,
            n_results=config.retrieval_results,
            threshold=config.similarity_threshold,
        )
        if not results:
            print("\nNo context found to answer the question.")
            continue

        context = "\n\n".join(doc["content"] for doc in results)
        print(f"\nContext:\n{context}")


if __name__ == "__main__":
    main()
