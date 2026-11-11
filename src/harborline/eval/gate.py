"""Compare metric values with the thresholds for one mode."""

from dataclasses import dataclass

from harborline.config import MetricThreshold


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    minimum: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    variant: str
    mode: str
    metrics: tuple[MetricResult, ...]

    @property
    def passed(self) -> bool:
        return all(metric.passed for metric in self.metrics)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(metric.name for metric in self.metrics if not metric.passed)


class IllustrativeScoresError(ValueError):
    """Raised when a file is explicitly not a real judge run."""


def apply_gate(
    scores: dict[str, float],
    thresholds: tuple[MetricThreshold, ...],
    *,
    variant: str,
    mode: str,
) -> GateResult:
    results: list[MetricResult] = []
    for threshold in thresholds:
        if mode not in threshold.modes:
            continue
        if threshold.name not in scores:
            results.append(
                MetricResult(
                    name=threshold.name, value=0.0, minimum=threshold.minimum, passed=False
                )
            )
            continue
        value = scores[threshold.name]
        results.append(
            MetricResult(
                name=threshold.name,
                value=value,
                minimum=threshold.minimum,
                passed=value + 1e-12 >= threshold.minimum,
            )
        )
    if not results:
        raise ValueError(f"no thresholds apply to mode {mode}")
    return GateResult(variant=variant, mode=mode, metrics=tuple(results))


def require_real_judge_file(payload: dict[str, object]) -> None:
    label = payload.get("label")
    real_run = payload.get("real_run")
    if real_run is not True or label == "illustrative_example_not_a_real_run":
        raise IllustrativeScoresError(
            "This judge-score file is illustrative. It has no measured scores. "
            "Live mode computes RAGAS judge metrics only when OPENAI_API_KEY is set."
        )
