import pytest
from pydantic import ValidationError

from llm_eval.results.tender_scout import EvaluationExport, compare_exports


def make_export(rows: list[tuple[str, bool, bool | None, int]]) -> EvaluationExport:
    return EvaluationExport.model_validate(
        {
            "goldenSetSize": len(rows),
            "questions": [
                {
                    "id": case,
                    "answerable": answerable,
                    "refused": refused,
                    "citations": relevant,
                    "relevantCitations": relevant,
                }
                for case, answerable, refused, relevant in rows
            ],
        }
    )


def test_compares_citation_and_refusal_by_id() -> None:
    baseline = make_export([("one", True, False, 1), ("two", False, False, 0)])
    candidate = make_export([("two", False, True, 0), ("one", True, False, 0)])
    report = compare_exports(baseline, candidate)
    assert report["baseline_passed"] == report["candidate_passed"] == 1
    assert report["regressions"] == ["one"]
    assert report["improvements"] == ["two"]


@pytest.mark.parametrize(
    "rows",
    [
        [("different", True, False, 1)],
        [("one", False, True, 0)],
        [("one", True, False, 1), ("one", True, False, 1)],
    ],
)
def test_rejects_incompatible_exports(rows: list[tuple[str, bool, bool, int]]) -> None:
    baseline = make_export([("one", True, False, 1)])
    with pytest.raises(ValueError):
        compare_exports(baseline, make_export(rows))


def test_rejects_retrieval_only_export() -> None:
    with pytest.raises(ValidationError):
        make_export([("one", True, None, 0)])
