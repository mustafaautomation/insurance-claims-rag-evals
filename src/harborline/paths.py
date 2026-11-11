"""Repository paths. The package is run from a checkout, not as a data-less wheel."""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def config_dir() -> Path:
    return repo_root() / "config"


def corpus_dir() -> Path:
    return data_dir() / "corpus"


def questions_path() -> Path:
    return data_dir() / "eval" / "questions.json"


def snapshot_path(variant: str) -> Path:
    return data_dir() / "snapshots" / f"{variant}.jsonl"


def judge_schema_path() -> Path:
    return data_dir() / "snapshots" / "judge_scores.illustrative.json"


def pipeline_config_path() -> Path:
    return config_dir() / "pipeline.yml"


def thresholds_path() -> Path:
    return config_dir() / "thresholds.yml"
