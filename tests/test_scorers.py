import pytest

from llm_eval.models import Sample
from llm_eval.providers.fake import ScriptedModel
from llm_eval.scorers.base import Score, Scorer
from llm_eval.scorers.exact import ExactMatch, RegexMatch
from llm_eval.scorers.judge import JudgeError, JudgeScorer
from llm_eval.scorers.registry import available_scorers, build_scorers


def make_sample(reference: str) -> Sample:
    return Sample(id="q1", input="What is the capital of Poland?", reference=reference)


def grade_json(grade: int) -> str:
    return f'{{"grade": {grade}, "reason": "Graded {grade}."}}'


def test_all_scorers_satisfy_protocol() -> None:
    scorers: list[Scorer] = [ExactMatch(), RegexMatch(), JudgeScorer(ScriptedModel([]))]

    assert [scorer.name for scorer in scorers] == ["exact", "regex", "judge"]


@pytest.mark.parametrize(
    ("answer", "reference", "expected"),
    [
        ("Warsaw", "Warsaw", 1.0),
        ("warsaw", "Warsaw", 1.0),
        ("  WARSAW \n", "Warsaw", 1.0),
        ("Warsaw.", "Warsaw", 1.0),
        ("Vatican   City", "Vatican City", 1.0),
        ("Warszawa", "warszawa", 1.0),
        ("Kraków", "Warsaw", 0.0),
        ("Warsaw, Poland", "Warsaw", 0.0),
        ("", "Warsaw", 0.0),
    ],
)
@pytest.mark.asyncio
async def test_exact_match(answer: str, reference: str, expected: float) -> None:
    score = await ExactMatch().score(make_sample(reference), answer)

    assert score.value == expected


@pytest.mark.asyncio
async def test_exact_match_can_keep_trailing_period() -> None:
    scorer = ExactMatch(ignore_trailing_period=False)

    score = await scorer.score(make_sample("Warsaw"), "Warsaw.")

    assert score.value == 0.0


@pytest.mark.asyncio
async def test_exact_match_explains_mismatch() -> None:
    score = await ExactMatch().score(make_sample("Warsaw"), "Kraków")

    assert score == Score("exact", 0.0, "expected 'Warsaw'")


@pytest.mark.parametrize(
    ("answer", "pattern", "expected"),
    [
        ("Warsaw", r"warsaw", 1.0),
        (" Warsaw ", r"Warsaw", 1.0),
        ("Warszawa", r"wars(aw|zawa)", 1.0),
        ("Lake Victoria", r"(lake )?victoria", 1.0),
        ("Victoria", r"(lake )?victoria", 1.0),
        ("Warsaw is the capital", r"Warsaw", 0.0),
        ("New Warsaw", r"Warsaw", 0.0),
        ("Krakow", r"wars(aw|zawa)", 0.0),
    ],
)
@pytest.mark.asyncio
async def test_regex_match(answer: str, pattern: str, expected: float) -> None:
    score = await RegexMatch().score(make_sample(pattern), answer)

    assert score.value == expected


@pytest.mark.asyncio
async def test_regex_match_rejects_invalid_pattern() -> None:
    with pytest.raises(ValueError, match="invalid regex reference"):
        await RegexMatch().score(make_sample("(unclosed"), "Warsaw")


@pytest.mark.parametrize(("grade", "expected"), [(1, 0.0), (3, 0.5), (5, 1.0)])
@pytest.mark.asyncio
async def test_judge_scorer_maps_grade_to_unit_range(
    grade: int, expected: float
) -> None:
    scorer = JudgeScorer(ScriptedModel([grade_json(grade)]))

    score = await scorer.score(make_sample("Warsaw"), "Warsaw")

    assert score == Score("judge", expected, f"Graded {grade}.")


@pytest.mark.asyncio
async def test_judge_scorer_sends_reference_and_answer() -> None:
    model = ScriptedModel([grade_json(5)])

    await JudgeScorer(model).score(make_sample("Warsaw"), "It is Warsaw")

    [(_, prompt)] = model.calls
    assert "What is the capital of Poland?" in prompt
    assert "Reference answer: Warsaw" in prompt
    assert "Answer to grade: It is Warsaw" in prompt


@pytest.mark.asyncio
async def test_judge_scorer_retries_after_out_of_range_grade() -> None:
    model = ScriptedModel([grade_json(7), grade_json(4)])

    score = await JudgeScorer(model).score(make_sample("Warsaw"), "Warsaw")

    assert score.value == 0.75
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_judge_scorer_raises_after_two_invalid_replies() -> None:
    model = ScriptedModel(["not json", "still not json"])

    with pytest.raises(JudgeError, match="JudgeGrade"):
        await JudgeScorer(model).score(make_sample("Warsaw"), "Warsaw")


def test_build_scorers_creates_scorers_in_order() -> None:
    scorers = build_scorers(["regex", "exact", "judge"], judge_model=ScriptedModel([]))

    assert [scorer.name for scorer in scorers] == ["regex", "exact", "judge"]


def test_build_scorers_rejects_unknown_name_listing_available() -> None:
    with pytest.raises(ValueError) as error:
        build_scorers(["exact", "bleu"])

    assert str(error.value) == "Unknown scorer 'bleu'. Available: exact, judge, regex."


def test_build_scorers_requires_model_for_judge() -> None:
    with pytest.raises(ValueError, match="needs a judge model"):
        build_scorers(["judge"])


def test_available_scorers_is_sorted() -> None:
    assert available_scorers() == ["exact", "judge", "regex"]
