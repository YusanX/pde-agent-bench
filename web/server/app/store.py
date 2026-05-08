import json
from functools import lru_cache
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "mock.json"


@lru_cache(maxsize=1)
def load() -> dict:
    with DATA_FILE.open(encoding="utf-8") as f:
        return json.load(f)
