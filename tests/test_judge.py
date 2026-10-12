import pytest

from llm_eval.judge import JudgeError, judge_once, judge_pair
from llm_eval.llm import ScriptedModel

A_WINS = '{"winner": "A", "reason": "More precise."}'
B_WINS = '{"winner": "B", "reason": "More precise."}'
TIE = '{"winner": "tie", "reason": "Equally good."}'
INVALID = "I think the first answer is better."


def test_judge_pair_returns_a_when_both_orders_agree() -> None:
    # Second call sees the answers swapped, so "B" there means answer_a.
    model = ScriptedModel([A_WINS, B_WINS])

    assert judge_pair(model, "What is 2+2?", "4", "5") == "A"


def test_judge_pair_returns_tie_when_orders_disagree() -> None:
    # Position bias: the judge always prefers the first answer.
    model = ScriptedModel([A_WINS, A_WINS])

    assert judge_pair(model, "What is 2+2?", "4", "5") == "tie"


def test_judge_pair_swaps_answers_in_second_prompt() -> None:
    model = ScriptedModel([A_WINS, B_WINS])

    judge_pair(model, "What is 2+2?", "first answer", "second answer")

    (_, original), (_, swapped) = model.calls
    assert original.index("first answer") < original.index("second answer")
    assert swapped.index("second answer") < swapped.index("first answer")


def test_judge_pair_keeps_tie_when_both_orders_tie() -> None:
    model = ScriptedModel([TIE, TIE])

    assert judge_pair(model, "What is 2+2?", "4", "four") == "tie"


def test_judge_once_retries_after_invalid_json() -> None:
    model = ScriptedModel([INVALID, B_WINS])

    assert judge_once(model, "What is 2+2?", "5", "4") == "B"
    assert len(model.calls) == 2


def test_judge_once_retries_after_unknown_winner() -> None:
    model = ScriptedModel(['{"winner": "C", "reason": "?"}', A_WINS])

    assert judge_once(model, "What is 2+2?", "4", "5") == "A"


def test_judge_once_raises_after_two_invalid_answers() -> None:
    model = ScriptedModel([INVALID, INVALID])

    with pytest.raises(JudgeError):
        judge_once(model, "What is 2+2?", "4", "5")
