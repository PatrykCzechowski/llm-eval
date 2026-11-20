"""Prepare a private case map and publishable inputs for the Tender Scout CI report."""

import argparse
import json
import os
import re
from pathlib import Path

from llm_eval.results.tender_scout import EvaluationExport, compare_exports

type DispatchInputs = dict[str, str]
CASE_ID = re.compile(r"case-[0-9]{4}")
MAX_PAYLOAD_BYTES = 65_535


def prepare_inputs(
    baseline: EvaluationExport, candidate: EvaluationExport
) -> tuple[DispatchInputs, dict[str, str]]:
    """Discard export metadata and replace source identifiers with report aliases."""
    compare_exports(baseline, candidate)
    aliases = {
        original: f"case-{number:04d}"
        for number, original in enumerate(sorted(baseline.by_id()), start=1)
    }

    def serialize(export: EvaluationExport) -> str:
        questions = [
            {**row.model_dump(by_alias=True), "id": aliases[original]}
            for original, row in sorted(export.by_id().items())
        ]
        return json.dumps(
            {"goldenSetSize": len(questions), "questions": questions},
            separators=(",", ":"),
        )

    inputs = {"baseline": serialize(baseline), "candidate": serialize(candidate)}
    if len(json.dumps(inputs, separators=(",", ":")).encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError("Pair exceeds the GitHub workflow input limit.")
    return inputs, {alias: original for original, alias in aliases.items()}


def summarize_event(event: Path, summary: Path) -> None:
    inputs = json.loads(event.read_text(encoding="utf-8"))["inputs"]
    if not isinstance(inputs, dict):
        raise TypeError("Expected workflow inputs.")
    baseline = EvaluationExport.model_validate_json(inputs["baseline"])
    candidate = EvaluationExport.model_validate_json(inputs["candidate"])
    if any(
        not CASE_ID.fullmatch(row.id)
        for export in (baseline, candidate)
        for row in export.questions
    ):
        raise ValueError(
            "Prepare the reports locally to replace source IDs with aliases."
        )
    report = compare_exports(baseline, candidate)
    sections = [
        "## Tender Scout A/B",
        "",
        f"- Baseline: **{report['baseline_passed']} / {report['cases']}**",
        f"- Candidate: **{report['candidate_passed']} / {report['cases']}**",
    ]
    for key, label in (
        ("regressions", "Regressions"),
        ("improvements", "Improvements"),
    ):
        aliases = report[key]
        assert isinstance(aliases, list)
        summary_items = ", ".join(f"`{alias}`" for alias in aliases)
        sections.append(f"- {label}: {summary_items or '0'}")
    with summary.open("a", encoding="utf-8") as file:
        file.write("\n".join(sections) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("baseline", type=Path)
    prepare.add_argument("candidate", type=Path)
    prepare.add_argument("case_map", type=Path, help="Local-only JSON mapping output")
    commands.add_parser("summarize-event")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            baseline = EvaluationExport.model_validate_json(args.baseline.read_bytes())
            candidate = EvaluationExport.model_validate_json(
                args.candidate.read_bytes()
            )
            inputs, case_map = prepare_inputs(baseline, candidate)
            case_map_fd = os.open(
                args.case_map, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(case_map_fd, "w", encoding="utf-8") as file:
                json.dump(case_map, file, indent=2, ensure_ascii=False)
                file.write("\n")
            print(json.dumps(inputs, separators=(",", ":")))
        else:
            summarize_event(
                Path(os.environ["GITHUB_EVENT_PATH"]),
                Path(os.environ["GITHUB_STEP_SUMMARY"]),
            )
    except (KeyError, OSError, TypeError, ValueError):
        # The GitHub repository is public. Never echo source input on validation failure.
        parser.exit(
            2,
            "Invalid Tender Scout input or output path. Use two complete exports with "
            "matching case IDs and labels, a new case-map path, and a pair that fits "
            "in the GitHub workflow input limit.\n",
        )


if __name__ == "__main__":
    main()
