from typing import List

from fastapi import APIRouter

from ..schemas import Figure
from ..store import load

router = APIRouter()


@router.get("/figures", response_model=List[Figure])
def list_figures() -> List[Figure]:
    return [Figure(**f) for f in load()["figures"]]
