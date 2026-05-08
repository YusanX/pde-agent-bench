from typing import List

from fastapi import APIRouter

from ..schemas import ModelMeta
from ..store import load

router = APIRouter()


@router.get("/models", response_model=List[ModelMeta])
def list_models() -> List[ModelMeta]:
    return [ModelMeta(**m) for m in load()["models"]]
