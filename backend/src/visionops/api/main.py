"""API entry point. Training and model loading do not run at API startup."""

from fastapi import FastAPI
from pydantic import BaseModel

from visionops.config import get_settings


class HealthResponse(BaseModel):
    status: str


def create_app() -> FastAPI:
    app = FastAPI(title=get_settings().app_name, version="0.1.0")

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Process liveness only; this does not assert database or model readiness."""
        return HealthResponse(status="ok")

    return app


app = create_app()
