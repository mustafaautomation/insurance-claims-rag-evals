"""Evaluation command line. Exit 0 when the gate passes, 1 when it fails, 2 on setup errors."""

import argparse
import sys
from pathlib import Path

from harborline.eval.gate import GateResult, IllustrativeScoresError
from harborline.eval.live import MissingApiKeyError
from harborline.eval.metrics import question_scores
from harborline.eval.report import VariantReport, write_reports
from harborline.eval.runner import gate_variant, record_all, record_variant, score_live
from harborline.models import VARIANTS, Variant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harborline-eval")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--variant", choices=VARIANTS, default="improved")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Score improved, no-guardrail, and bad-retriever. The exit code follows improved.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Re-run the offline pipeline and overwrite the snapshot JSONL before scoring.",
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.mode == "live":
            return _live(args.variant, record=args.record, report_dir=args.report_dir)
        return _offline(
            args.variant,
            compare=args.compare,
            record=args.record,
            report_dir=args.report_dir,
        )
    except MissingApiKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (FileNotFoundError, IllustrativeScoresError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _live(variant: Variant, *, record: bool, report_dir: Path | None) -> int:
    gate, _scores, rows = score_live(variant, record=record)
    _print_gate(gate)
    if report_dir is not None:
        write_reports(
            report_dir,
            {variant: VariantReport(gate=gate, questions=tuple(question_scores(rows)))},
            llm_judged=True,
        )
    return 0 if gate.passed else 1


def _offline(
    variant: Variant,
    *,
    compare: bool,
    record: bool,
    report_dir: Path | None,
) -> int:
    if record:
        if compare:
            record_all()
        else:
            record_variant(variant)
    if compare:
        reports: dict[str, VariantReport] = {}
        for name in VARIANTS:
            gate, _scores, rows = gate_variant(name)
            reports[name] = VariantReport(gate=gate, questions=tuple(question_scores(rows)))
        if report_dir is not None:
            write_reports(report_dir, reports, llm_judged=False)
        _print_gate(reports["improved"].gate)
        return 0 if reports["improved"].gate.passed else 1
    gate, _scores, rows = gate_variant(variant)
    _print_gate(gate)
    if report_dir is not None:
        write_reports(
            report_dir,
            {variant: VariantReport(gate=gate, questions=tuple(question_scores(rows)))},
            llm_judged=False,
        )
    return 0 if gate.passed else 1


def _print_gate(gate: GateResult) -> None:
    print(f"variant={gate.variant} mode={gate.mode} passed={str(gate.passed).lower()}")
    for metric in gate.metrics:
        flag = "pass" if metric.passed else "fail"
        print(f"{metric.name}={metric.value:.3f} minimum={metric.minimum:.3f} {flag}")


if __name__ == "__main__":
    raise SystemExit(main())
