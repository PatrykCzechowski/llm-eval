"""Common interface of all scoring methods."""

from dataclasses import dataclass
from typing import Protocol

from llm_eval.models import Sample


@dataclass(frozen=True)
class Score:
    """Result of one scorer for one answer; ``value`` is between 0.0 and 1.0."""

    name: str
    value: float
    detail: str = ""


class Scorer(Protocol):
    """Scores a model answer against a dataset sample."""

    name: str

    async def score(self, sample: Sample, answer: str) -> Score: ...
