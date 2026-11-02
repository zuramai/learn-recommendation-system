import json
from pathlib import Path

BOOK = Path(__file__).resolve().parents[2]


def results(name: str) -> dict:
    return json.loads((BOOK / "results" / f"{name}.json").read_text())
