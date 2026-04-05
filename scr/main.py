import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scr.config import PipelineConfig
else:
    from .config import PipelineConfig


def main():
    if __package__ is None or __package__ == "":
        from scr.pipeline import build_pipeline, rag_simple
    else:
        from .pipeline import build_pipeline, rag_simple

    config = PipelineConfig()
    print("Building PDF RAG pipeline...")
    pipeline = build_pipeline(config)
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

        answer = rag_simple(
            query=query,
            llm=pipeline["llm"],
            retriever=pipeline["rag_retriever"],
            n_results=config.retrieval_results,
        )
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
