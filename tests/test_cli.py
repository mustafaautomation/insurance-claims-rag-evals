"""Command-line exit codes for the offline gate and the live-mode key check."""

from pathlib import Path

import pytest

from harborline.eval.cli import main


def test_offline_improved_exits_zero(tmp_path: Path) -> None:
    assert main(["--mode", "offline", "--variant", "improved", "--report-dir", str(tmp_path)]) == 0
    assert (tmp_path / "eval_report.md").exists()
    assert (tmp_path / "eval_report.html").exists()


def test_bad_retriever_cli_exits_nonzero() -> None:
    assert main(["--mode", "offline", "--variant", "bad_retriever"]) == 1


def test_no_guardrail_cli_exits_nonzero() -> None:
    assert main(["--mode", "offline", "--variant", "no_guardrail"]) == 1


def test_live_mode_exits_when_the_key_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--mode", "live", "--variant", "improved"]) == 2
