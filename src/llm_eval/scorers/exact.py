"""Deterministic scorers: exact text match and regular expression match."""

import re

from llm_eval.models import Sample
from llm_eval.scorers.base import Score


class ExactMatch:
    """Answer equals the reference after normalization (case, whitespace, final dot)."""

    name = "exact"

    def __init__(self, ignore_trailing_period: bool = True) -> None:
        self.ignore_trailing_period = ignore_trailing_period

    def normalize(self, text: str) -> str:
        """Casefold, collapse whitespace and optionally drop one final period."""
        normalized = " ".join(text.casefold().split())
        if self.ignore_trailing_period:
            normalized = normalized.removesuffix(".").rstrip()
        return normalized

    async def score(self, sample: Sample, answer: str) -> Score:
        """Return 1.0 when the normalized texts are equal, else 0.0."""
        if self.normalize(answer) == self.normalize(sample.reference):
            return Score(self.name, 1.0)
        return Score(self.name, 0.0, f"expected {sample.reference!r}")


class RegexMatch:
    """The whole answer matches the reference used as a case-insensitive pattern."""

    name = "regex"

    async def score(self, sample: Sample, answer: str) -> Score:
        """Return 1.0 on a full match; raise ValueError for an invalid pattern."""
        try:
            pattern = re.compile(sample.reference, flags=re.IGNORECASE)
        except re.error as error:
            raise ValueError(
                f"Sample {sample.id!r} has an invalid regex reference: {error}"
            ) from error
        if pattern.fullmatch(answer.strip()) is not None:
            return Score(self.name, 1.0)
        return Score(self.name, 0.0, f"no full match for {sample.reference!r}")
