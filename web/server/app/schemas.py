from typing import List, Dict, Optional, Literal
from pydantic import BaseModel


# ── Stats / Hero ─────────────────────────────────────────────────────────────
class Stats(BaseModel):
    totalCases: int
    pdeTypes: int
    backends: int
    modelsEvaluated: int
    agents: int
    version: str


# ── Leaderboard ──────────────────────────────────────────────────────────────
class LeaderboardRow(BaseModel):
    id: int
    model: str
    provider: str
    agent: Optional[str] = None
    backend: Literal["dolfinx", "firedrake", "dealii"]
    passRate: float
    accuracyGate: Optional[float] = None
    timeGate: Optional[float] = None
    l2Error: Optional[float] = None
    costPer1K: Optional[float] = None
    date: Optional[str] = None
    pdeScores: Dict[str, float]


# ── PDE Types ────────────────────────────────────────────────────────────────
class PdeType(BaseModel):
    id: str
    name: str
    abbr: str
    category: str
    color: str
    bgColor: str
    cases: int
    equation: str
    description: str
    backends: List[str]
    backendCases: Dict[str, Optional[int]]


# ── Models catalog ───────────────────────────────────────────────────────────
class ModelMeta(BaseModel):
    id: str
    displayName: str
    provider: str
    family: Literal["LLM", "Agent"]
    paramSize: Optional[str] = None
    notes: Optional[str] = None


# ── Authors ──────────────────────────────────────────────────────────────────
class Author(BaseModel):
    name: str
    affiliation: str
    homepage: Optional[str] = None
    isCorresponding: bool = False
    isEqual: bool = False


class AuthorsResponse(BaseModel):
    authors: List[Author]
    affiliations: List[str]


# ── Findings ─────────────────────────────────────────────────────────────────
class Finding(BaseModel):
    n: int
    title: str
    body: str
    icon: str


# ── Figures ──────────────────────────────────────────────────────────────────
class Figure(BaseModel):
    id: str
    src: str
    thumb: str
    caption: str
    section: str


# ── Citation ─────────────────────────────────────────────────────────────────
class Citation(BaseModel):
    title: str
    venue: str
    year: int
    abstract: str
    bibtex: str
