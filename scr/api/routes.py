from fastapi import APIRouter, Depends, HTTPException, Request

from scr.api.schemas import (
    MessageResponse,
    PipelineStatusResponse,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from scr.services.pipeline_service import PipelineService

router = APIRouter()


def get_pipeline_service(request: Request) -> PipelineService:
    return request.app.state.pipeline_service


@router.get("/health", response_model=MessageResponse)
def health_check():
    return MessageResponse(message="API is running.")


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
def pipeline_status(service: PipelineService = Depends(get_pipeline_service)):
    return PipelineStatusResponse(**service.get_status())


@router.post("/pipeline/build", response_model=PipelineStatusResponse)
def build_pipeline_endpoint(service: PipelineService = Depends(get_pipeline_service)):
    try:
        service.build()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PipelineStatusResponse(**service.get_status())


@router.post("/query", response_model=QueryResponse)
def query_pipeline(
    payload: QueryRequest,
    service: PipelineService = Depends(get_pipeline_service),
):
    try:
        result = service.query(query=payload.query, n_results=payload.n_results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(**result)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_documents(
    payload: RetrieveRequest,
    service: PipelineService = Depends(get_pipeline_service),
):
    try:
        results = service.retrieve(
            query=payload.query,
            n_results=payload.n_results,
            threshold=payload.threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RetrieveResponse(query=payload.query, results=results)
