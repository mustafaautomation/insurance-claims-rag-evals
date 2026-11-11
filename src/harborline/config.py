"""Load pipeline settings and metric thresholds."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from harborline.paths import pipeline_config_path, thresholds_path


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int
    chunk_size: int
    chunk_overlap: int
    embedding_dim: int
    min_retrieval_score: float
    chat_model: str
    judge_model: str
    embedding_model: str


@dataclass(frozen=True)
class MetricThreshold:
    name: str
    minimum: float
    modes: tuple[str, ...]


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    loaded = _mapping(_read_yaml(path or pipeline_config_path()))
    return PipelineConfig(
        top_k=_int(loaded, "top_k"),
        chunk_size=_int(loaded, "chunk_size"),
        chunk_overlap=_int(loaded, "chunk_overlap"),
        embedding_dim=_int(loaded, "embedding_dim"),
        min_retrieval_score=_float(loaded, "min_retrieval_score"),
        chat_model=_string(loaded, "chat_model"),
        judge_model=_string(loaded, "judge_model"),
        embedding_model=_string(loaded, "embedding_model"),
    )


def load_thresholds(path: Path | None = None) -> tuple[MetricThreshold, ...]:
    loaded = _mapping(_read_yaml(path or thresholds_path()))
    metrics = loaded.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("thresholds.yml needs a metrics mapping")
    parsed: list[MetricThreshold] = []
    for name, raw in metrics.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ValueError("each metric threshold needs a string name and a mapping")
        item = _mapping(raw)
        modes_raw = item.get("modes")
        if not isinstance(modes_raw, list) or not modes_raw:
            raise ValueError(f"{name} needs a non-empty modes list")
        modes: list[str] = []
        for mode in modes_raw:
            if not isinstance(mode, str):
                raise ValueError(f"{name} modes must be strings")
            modes.append(mode)
        parsed.append(
            MetricThreshold(name=name, minimum=_float(item, "minimum"), modes=tuple(modes))
        )
    return tuple(parsed)


def _read_yaml(path: Path) -> object:
    value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value


def _mapping(value: object) -> dict[object, object]:
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping")
    return {key: item for key, item in value.items()}


def _string(mapping: dict[object, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _int(mapping: dict[object, object], key: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _float(mapping: dict[object, object], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)
