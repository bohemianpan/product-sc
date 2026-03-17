"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.api.simulation_routes import router as simulation_router

app = FastAPI(
    title="MiroFish",
    description="Multi-agent simulation prediction engine — MVP",
    version="0.2.0",
)

app.include_router(router, prefix="/api/v1")
app.include_router(simulation_router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
