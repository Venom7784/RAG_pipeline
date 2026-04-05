from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scr.api.routes import router
from scr.config import PipelineConfig
from scr.services.pipeline_service import PipelineService


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = PipelineService(config=PipelineConfig())
    app.state.pipeline_service = service
    try:
        service.build()
    except Exception as exc:
        service.last_error = str(exc)
    yield


def create_app():
    app = FastAPI(
        title="PDF RAG Backend",
        description="FastAPI backend for the PDF RAG pipeline.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1", tags=["rag"])
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "scr.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
