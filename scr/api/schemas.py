from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask the RAG system.")
    n_results: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of chunks to retrieve.",
    )


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query used for retrieval.")
    n_results: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of chunks to retrieve.",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional similarity threshold.",
    )


class RetrievedDocument(BaseModel):
    content: str
    metadata: dict
    similarity_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    results: list[RetrievedDocument]


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedDocument]


class PipelineStatusResponse(BaseModel):
    ready: bool
    collection_name: str
    pdf_directory: str
    persist_directory: str
    retrieval_results: int
    similarity_threshold: float
    vector_count: int
    last_built_at: str | None
    last_error: str | None


class MessageResponse(BaseModel):
    message: str
