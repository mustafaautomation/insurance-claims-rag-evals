"""Read and write JSONL recordings of pipeline outputs."""

import json
from pathlib import Path

from harborline.models import Row, row_from_json
from harborline.paths import snapshot_path


def write_snapshot(variant: str, rows: list[Row], path: Path | None = None) -> Path:
    target = path or snapshot_path(variant)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row.to_json(), ensure_ascii=False, sort_keys=True) for row in rows]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def read_snapshot(variant: str, path: Path | None = None) -> list[Row]:
    target = path or snapshot_path(variant)
    if not target.exists():
        raise FileNotFoundError(
            f"missing snapshot {target}. Run harborline-eval --record --variant {variant}."
        )
    rows: list[Row] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        loaded: object = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"{target}:{line_number} is not a JSON object")
        payload: dict[str, object] = {}
        for key, value in loaded.items():
            if not isinstance(key, str):
                raise ValueError(f"{target}:{line_number} has a non-string key")
            payload[key] = value
        rows.append(row_from_json(payload, variant))
    if not rows:
        raise ValueError(f"{target} is empty")
    return rows
