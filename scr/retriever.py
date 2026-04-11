from .embeddings import EmbeddingManager
from .query_rewriter import QueryRewriter
from .vector_store import VectorStore


class RAGRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        query_rewriter: QueryRewriter | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.query_rewriter = query_rewriter or QueryRewriter()

    def rewrite_query(self, query: str) -> str:
        return self.query_rewriter.rewrite(query)

    def retriever(self, query: str, n_results: int = 5, threshold: float = 0.5):
        retrieval_query = self.rewrite_query(query)
        query_embedding = self.embedding_manager.generate_embeddings([retrieval_query])[0]
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            n_results=n_results,
        )

        retrieved_docs = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, metadata, distance in zip(documents, metadatas, distances):
            similarity_score = 1 - distance
            if similarity_score >= threshold:
                retrieved_docs.append(
                    {
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                    }
                )

        return retrieved_docs
