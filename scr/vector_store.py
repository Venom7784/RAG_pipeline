import os
import uuid
from typing import Any

import chromadb
import numpy as np


class VectorStore:
    def __init__(
        self,
        collection: str = "pdf_documents",
        persist_directory: str = "data/vector_store",
    ):
        self.collection_name = collection
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents, embeddings):
        if not documents:
            raise ValueError("No documents provided.")

        if len(documents) != len(embeddings):
            raise ValueError("The number of documents and embeddings must match.")

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )

        print(f"Added {len(documents)} documents to collection '{self.collection_name}'.")

    def similarity_search(self, query_embedding, n_results: int = 5) -> dict[str, Any]:
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results

    def count(self) -> int:
        return self.collection.count()
