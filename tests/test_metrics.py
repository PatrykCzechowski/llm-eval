import pytest

from llm_eval.metrics import agreement


def test_agreement_lengths_raises_value_error() -> None:
    judge = ["A", "B", "tie"]
    humans = ["A", "B"]

    with pytest.raises(ValueError):
        agreement(judge, humans)


def test_agreement_empty_lists_raises_value_error() -> None:
    judge: list[str] = []
    humans: list[str] = []

    with pytest.raises(ValueError):
        agreement(judge, humans)


def test_agreement_full_match_returns_one() -> None:
    judge = ["A", "B", "tie"]
    humans = ["A", "B", "tie"]

    assert agreement(judge, humans) == 1.0


def test_agreement_half_match_returns_half() -> None:
    judge = ["A", "A", "tie", "B"]
    humans = ["A", "B", "B", "B"]

    assert agreement(judge, humans) == 0.5
