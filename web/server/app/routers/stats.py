from fastapi import APIRouter

from ..schemas import Stats
from ..store import load

router = APIRouter()


@router.get("/stats", response_model=Stats)
def get_stats() -> Stats:
    return Stats(**load()["stats"])
