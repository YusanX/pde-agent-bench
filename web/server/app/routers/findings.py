from typing import List

from fastapi import APIRouter

from ..schemas import Finding
from ..store import load

router = APIRouter()


@router.get("/findings", response_model=List[Finding])
def list_findings() -> List[Finding]:
    return [Finding(**f) for f in load()["findings"]]
