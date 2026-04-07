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
        embedding_dimension: int | None = None,
        embedding_model_name: str | None = None,
    ):
        self.collection_name = collection
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        self.embedding_model_name = embedding_model_name
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            expected_metadata = self._expected_metadata()

            try:
                collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=expected_metadata,
                )

            collection_metadata = collection.metadata or {}
            if self._metadata_mismatch(collection_metadata):
                print(
                    f"Embedding configuration changed for collection "
                    f"'{self.collection_name}'. Recreating vector store."
                )
                self.client.delete_collection(name=self.collection_name)
                collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=expected_metadata,
                )

            self.collection = collection
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def _expected_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"hnsw:space": "cosine"}
        if self.embedding_dimension is not None:
            metadata["embedding_dimension"] = self.embedding_dimension
        if self.embedding_model_name is not None:
            metadata["embedding_model_name"] = self.embedding_model_name
        return metadata

    def _metadata_mismatch(self, metadata: dict[str, Any]) -> bool:
        stored_dimension = metadata.get("embedding_dimension")
        stored_model_name = metadata.get("embedding_model_name")

        if self.embedding_dimension is not None:
            if stored_dimension is None or stored_dimension != self.embedding_dimension:
                return True

        if self.embedding_model_name is not None:
            if stored_model_name is None or stored_model_name != self.embedding_model_name:
                return True

        return False

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
