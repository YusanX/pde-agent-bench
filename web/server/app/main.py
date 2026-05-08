from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    authors,
    citation,
    figures,
    findings,
    leaderboard,
    models,
    pde_types,
    stats,
)

app = FastAPI(
    title="PDEAgent-Bench API",
    description="Mock backend for the PDEAgent-Bench showcase site.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


for r in (stats, leaderboard, models, pde_types, findings, authors, figures, citation):
    app.include_router(r.router, prefix="/api", tags=[r.__name__.split(".")[-1]])
