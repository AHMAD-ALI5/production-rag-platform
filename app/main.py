
from fastapi import FastAPI

from app.core.config import settings
from app.api.documents import router as documents_router

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description="PDF question answering with grounded citations.",
)

app.include_router(documents_router)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "status": "running",
        "version": "0.2.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
