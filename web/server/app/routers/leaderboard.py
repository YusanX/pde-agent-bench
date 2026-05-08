from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas import LeaderboardRow
from ..store import load

router = APIRouter()

SORT_KEYS = {"passRate", "accuracyGate", "timeGate", "l2Error", "costPer1K"}


@router.get("/leaderboard", response_model=List[LeaderboardRow])
def list_leaderboard(
    backend: Optional[str] = Query(None, pattern="^(all|dolfinx|firedrake|dealii)$"),
    sort: str = Query("passRate"),
    limit: Optional[int] = Query(None, gt=0, le=100),
) -> List[LeaderboardRow]:
    if sort not in SORT_KEYS:
        raise HTTPException(400, detail=f"sort must be one of {sorted(SORT_KEYS)}")
    rows = load()["leaderboard"]
    if backend and backend != "all":
        rows = [r for r in rows if r["backend"] == backend]
    reverse = sort != "l2Error" and sort != "costPer1K"
    rows = sorted(rows, key=lambda r: r[sort], reverse=reverse)
    if limit:
        rows = rows[:limit]
    return [LeaderboardRow(**r) for r in rows]


@router.get("/leaderboard/{model_id}", response_model=LeaderboardRow)
def get_leaderboard_row(model_id: int) -> LeaderboardRow:
    for r in load()["leaderboard"]:
        if r["id"] == model_id:
            return LeaderboardRow(**r)
    raise HTTPException(404, detail=f"row id {model_id} not found")
