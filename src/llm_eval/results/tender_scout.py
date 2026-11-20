"""Compare paired per-question outcomes from two Tender Scout evaluation JSON exports."""

import argparse
import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

NonnegativeInt = Annotated[int, Field(ge=0)]


class QuestionResult(BaseModel):
    model_config = ConfigDict(strict=True)

    id: Annotated[str, Field(min_length=1)]
    answerable: bool
    refused: bool
    citations: NonnegativeInt
    relevant_citations: NonnegativeInt = Field(alias="relevantCitations")

    def passed(self) -> bool:
        if self.answerable:
            return not self.refused and self.relevant_citations > 0
        return self.refused


class EvaluationExport(BaseModel):
    model_config = ConfigDict(strict=True)

    golden_set_size: int = Field(alias="goldenSetSize", gt=0)
    questions: list[QuestionResult]

    def by_id(self) -> dict[str, QuestionResult]:
        indexed = {question.id: question for question in self.questions}
        if len(indexed) != len(self.questions) or len(indexed) != self.golden_set_size:
            raise ValueError("Question IDs must be unique and match goldenSetSize.")
        if any(
            question.relevant_citations > question.citations
            for question in self.questions
        ):
            raise ValueError(
                "Relevant citation count exceeds the total citation count."
            )
        return indexed


def compare_passes(
    baseline: dict[str, bool], candidate: dict[str, bool]
) -> dict[str, object]:
    """Count paired outcomes and list the IDs that changed."""
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError(
            "Both runs must contain the same nonempty set of question IDs."
        )
    return {
        "cases": len(baseline),
        "baseline_passed": sum(baseline.values()),
        "candidate_passed": sum(candidate.values()),
        "regressions": sorted(
            case for case in baseline if baseline[case] and not candidate[case]
        ),
        "improvements": sorted(
            case for case in baseline if not baseline[case] and candidate[case]
        ),
    }


def compare_exports(
    baseline: EvaluationExport, candidate: EvaluationExport
) -> dict[str, object]:
    base_rows, candidate_rows = baseline.by_id(), candidate.by_id()
    base_labels = {case: row.answerable for case, row in base_rows.items()}
    candidate_labels = {case: row.answerable for case, row in candidate_rows.items()}
    if base_labels != candidate_labels:
        raise ValueError("Question IDs and answerable labels must match between runs.")
    return {
        "criterion": "Answered with a relevant citation, or refused when no answer existed.",
        **compare_passes(
            {case: row.passed() for case, row in base_rows.items()},
            {case: row.passed() for case, row in candidate_rows.items()},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline", type=Path, help="Tender Scout evaluation JSON for version A"
    )
    parser.add_argument(
        "candidate", type=Path, help="Tender Scout evaluation JSON for version B"
    )
    args = parser.parse_args()

    try:
        baseline = EvaluationExport.model_validate_json(args.baseline.read_bytes())
        candidate = EvaluationExport.model_validate_json(args.candidate.read_bytes())
        report = compare_exports(baseline, candidate)
    except ValidationError as exc:
        # Validation details can contain the original input.
        problems = ", ".join(
            ".".join(map(str, error["loc"]))
            for error in exc.errors(include_input=False)
        )
        parser.error(f"Invalid or incomplete evaluation export ({problems}).")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
