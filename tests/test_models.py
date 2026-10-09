import dataclasses

import pytest
from pydantic import ValidationError

from llm_eval.models import AppVersion, JudgeVerdict, Question


def test_judge_verdict_parses_valid_json() -> None:
    verdict = JudgeVerdict.model_validate_json(
        '{"winner": "A", "reason": "More precise."}'
    )

    assert verdict.winner == "A"
    assert verdict.reason == "More precise."


def test_judge_verdict_rejects_unknown_winner() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate_json('{"winner": "C", "reason": "More precise."}')


def test_judge_verdict_rejects_empty_reason() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate_json('{"winner": "tie", "reason": ""}')


def test_question_is_frozen() -> None:
    question = Question(id="q1", text="What is 2+2?", category="math", split="test")

    with pytest.raises(dataclasses.FrozenInstanceError):
        question.text = "changed"  # type: ignore[misc]


def test_app_version_default_temperature_is_zero() -> None:
    app = AppVersion(name="v1", system_prompt="Be helpful.", model="llama3.2")

    assert app.temperature == 0.0


def test_app_version_accepts_custom_temperature() -> None:
    app = AppVersion(
        name="v1", system_prompt="Be helpful.", model="llama3.2", temperature=0.7
    )

    assert app.temperature == 0.7
