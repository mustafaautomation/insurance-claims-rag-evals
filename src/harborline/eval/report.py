"""Markdown and HTML evaluation reports. Numbers come from the scored rows."""

from dataclasses import dataclass
from html import escape
from pathlib import Path

from harborline.eval.gate import GateResult
from harborline.eval.metrics import QuestionScore


@dataclass(frozen=True)
class VariantReport:
    gate: GateResult
    questions: tuple[QuestionScore, ...]


OFFLINE_NOTE = (
    "LLM-judged RAGAS metrics were not run for this report. "
    "No API key was available, so faithfulness, answer relevancy, context precision, "
    "and context recall are not given numbers here. "
    "The scores below were produced by the deterministic offline generator, "
    "retrieval checks, and RAGAS non-LLM metrics."
)


def write_reports(directory: Path, variants: dict[str, VariantReport], *, llm_judged: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(variants, llm_judged=llm_judged)
    html = render_html(variants, llm_judged=llm_judged)
    (directory / "eval_report.md").write_text(markdown, encoding="utf-8")
    (directory / "eval_report.html").write_text(html, encoding="utf-8")


def render_markdown(variants: dict[str, VariantReport], *, llm_judged: bool) -> str:
    if "improved" in variants and "no_guardrail" in variants and "bad_retriever" in variants:
        return _compare_markdown(variants, llm_judged=llm_judged)
    return _single_markdown(variants, llm_judged=llm_judged)


def _single_markdown(variants: dict[str, VariantReport], *, llm_judged: bool) -> str:
    name, report = next(iter(variants.items()))
    lines = [
        "# Harborline Mutual evaluation report",
        "",
        _note(llm_judged),
        "",
        f"Variant: {name}. Mode: {report.gate.mode}. Passed: {str(report.gate.passed).lower()}.",
        "",
        "| Metric | Score | Minimum | Gate |",
        "| --- | ---: | ---: | --- |",
    ]
    for metric in report.gate.metrics:
        status = "pass" if metric.passed else "fail"
        lines.append(f"| {metric.name} | {metric.value:.3f} | {metric.minimum:.3f} | {status} |")
    lines.extend(["", "## Per-question failures", ""])
    failures = [item for item in report.questions if _failed(item)]
    if not failures:
        lines.append("No per-question failures.")
    else:
        lines.extend(_failure_lines(failures))
    lines.append("")
    return "\n".join(lines)


def _compare_markdown(variants: dict[str, VariantReport], *, llm_judged: bool) -> str:
    improved = variants["improved"]
    lines: list[str] = [
        "# Harborline Mutual evaluation report",
        "",
        _note(llm_judged),
        "",
        "The improved row is the one the offline gate accepts or rejects. "
        "The no-guardrail prompt drops the approval, dispute, and out-of-scope rules. "
        "The bad retriever returns the lowest-scoring policy pages.",
        "",
        "## Metric table",
        "",
        _metric_table(variants),
        "",
        "## Before and after",
        "",
        _deltas(variants),
        "",
        "## Improved per-question failures",
        "",
    ]
    failures = [item for item in improved.questions if _failed(item)]
    if not failures:
        lines.append("No per-question failures on the improved variant.")
    else:
        lines.extend(_failure_lines(failures))
    lines.extend(["", "## Baseline failure counts", ""])
    for name in ("no_guardrail", "bad_retriever"):
        count = sum(1 for item in variants[name].questions if _failed(item))
        lines.append(
            f"- {name}: {count} of {len(variants[name].questions)} questions failed a check."
        )
    example = next((item for item in variants["bad_retriever"].questions if _failed(item)), None)
    if example is not None:
        lines.extend(
            [
                "",
                "## Example failure from the bad retriever",
                "",
                f"- id: {example.id}",
                f"- question: {example.question}",
                f"- expected behavior: {example.expected_behavior}",
                f"- behavior ok: {str(example.behavior_ok).lower()}",
                f"- retrieved docs: {', '.join(example.retrieved_doc_ids) or '(none)'}",
                f"- response: {example.response[:400]}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_html(variants: dict[str, VariantReport], *, llm_judged: bool) -> str:
    body = render_markdown(variants, llm_judged=llm_judged)
    rows: list[str] = []
    table_buffer: list[str] = []
    list_buffer: list[str] = []

    def flush_table() -> None:
        if not table_buffer:
            return
        rows.append(_html_table(table_buffer))
        table_buffer.clear()

    def flush_list() -> None:
        if not list_buffer:
            return
        rows.append("<ul>" + "".join(list_buffer) + "</ul>")
        list_buffer.clear()

    for line in body.splitlines():
        if line.startswith("|"):
            flush_list()
            table_buffer.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            flush_list()
            rows.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_list()
            rows.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("- "):
            list_buffer.append(f"<li>{escape(line[2:])}</li>")
        elif line == "":
            continue
        else:
            flush_list()
            rows.append(f"<p>{escape(line)}</p>")
    flush_table()
    flush_list()
    style = (
        "body{font-family:Georgia,serif;max-width:980px;margin:2rem auto;padding:0 1rem;"
        "color:#1c1917;background:#fafaf9;line-height:1.45}"
        "h1{font-size:1.7rem}h2{font-size:1.2rem;margin-top:1.8rem}"
        "table{border-collapse:collapse;width:100%;margin:0.6rem 0 1rem;background:white}"
        "th,td{border:1px solid #d6d3d1;padding:0.35rem 0.5rem;text-align:left;font-size:0.92rem}"
        "th{background:#e7e5e4}li{margin:0.25rem 0 0.25rem 1.2rem}"
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        "<title>Harborline Mutual evaluation report</title>"
        f"<style>{style}</style></head><body>{''.join(rows)}</body></html>\n"
    )


def _html_table(lines: list[str]) -> str:
    parsed = [
        _split_row(line)
        for line in lines
        if not set(line.replace("|", "").strip()) <= {"-", ":", " "}
    ]
    if not parsed:
        return ""
    head = "".join(f"<th>{escape(cell)}</th>" for cell in parsed[0])
    body = []
    for cells in parsed[1:]:
        body.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _note(llm_judged: bool) -> str:
    if llm_judged:
        return "This report includes live RAGAS judge metrics from an OpenAI run."
    return OFFLINE_NOTE


def _metric_table(variants: dict[str, VariantReport]) -> str:
    header = "| Metric | Minimum | Improved | No-guardrail prompt | Bad retriever | Improved gate |"
    rule = "| --- | ---: | ---: | ---: | ---: | --- |"
    lines = [header, rule]
    by_name = {
        name: {variant: _value(report, name) for variant, report in variants.items()}
        for name in [metric.name for metric in variants["improved"].gate.metrics]
    }
    for metric in variants["improved"].gate.metrics:
        values = by_name[metric.name]
        status = "pass" if metric.passed else "fail"
        lines.append(
            f"| {metric.name} | {metric.minimum:.3f} | {values['improved']:.3f} | "
            f"{values['no_guardrail']:.3f} | {values['bad_retriever']:.3f} | {status} |"
        )
    return "\n".join(lines)


def _deltas(variants: dict[str, VariantReport]) -> str:
    lines = [
        "| Metric | Improved minus no-guardrail | Improved minus bad retriever |",
        "| --- | ---: | ---: |",
    ]
    for metric in variants["improved"].gate.metrics:
        improved = _value(variants["improved"], metric.name)
        plain = _value(variants["no_guardrail"], metric.name)
        bad = _value(variants["bad_retriever"], metric.name)
        lines.append(f"| {metric.name} | {improved - plain:+.3f} | {improved - bad:+.3f} |")
    return "\n".join(lines)


def _failure_lines(failures: list[QuestionScore]) -> list[str]:
    lines: list[str] = []
    for item in failures:
        presence = "n/a" if item.string_presence is None else f"{item.string_presence:.3f}"
        hit = "n/a" if item.hit is None else str(item.hit).lower()
        lines.append(
            f"- {item.id}: behavior_ok={str(item.behavior_ok).lower()} "
            f"string_presence={presence} hit={hit} "
            f"retrieved={','.join(item.retrieved_doc_ids)}"
        )
    return lines


def _value(report: VariantReport, name: str) -> float:
    for metric in report.gate.metrics:
        if metric.name == name:
            return metric.value
    raise KeyError(name)


def _failed(item: QuestionScore) -> bool:
    if not item.behavior_ok:
        return True
    if item.string_presence is not None and item.string_presence < 1.0:
        return True
    return item.hit is False
