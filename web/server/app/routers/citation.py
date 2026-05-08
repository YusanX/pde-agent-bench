from fastapi import APIRouter

from ..schemas import Citation
from ..store import load

router = APIRouter()


@router.get("/citation", response_model=Citation)
def get_citation() -> Citation:
    return Citation(**load()["citation"])
