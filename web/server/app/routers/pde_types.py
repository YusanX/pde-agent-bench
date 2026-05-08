from typing import List

from fastapi import APIRouter

from ..schemas import PdeType
from ..store import load

router = APIRouter()


@router.get("/pde-types", response_model=List[PdeType])
def list_pde_types() -> List[PdeType]:
    return [PdeType(**p) for p in load()["pdeTypes"]]
