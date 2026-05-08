from fastapi import APIRouter

from ..schemas import AuthorsResponse
from ..store import load

router = APIRouter()


@router.get("/authors", response_model=AuthorsResponse)
def get_authors() -> AuthorsResponse:
    return AuthorsResponse(**load()["authors"])
