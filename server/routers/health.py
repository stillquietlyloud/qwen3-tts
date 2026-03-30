"""
Health / readiness endpoints.

GET /health           – liveness probe (always 200 once the process is up)
GET /health/ready     – readiness probe (200 only when model is loaded)
GET /v1/models        – list the loaded model (OpenAI-compatible)
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..model_manager import manager

router = APIRouter(tags=["health"])

_start_time = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model: str | None
    model_type: str | None
    model_loaded: bool


@router.get("/health", response_model=HealthResponse)
def liveness() -> HealthResponse:
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
        model=manager.model_name,
        model_type=manager.model_type,
        model_loaded=manager.is_loaded,
    )


@router.get("/health/ready", response_model=HealthResponse)
def readiness() -> HealthResponse:
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return liveness()


# ── OpenAI-compatible model list ──────────────────────────────────────────────

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "qwen"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


@router.get("/v1/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    models = []
    if manager.model_name:
        models.append(ModelObject(id=manager.model_name))
    return ModelListResponse(data=models)
