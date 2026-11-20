import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tender_scout_ci.py"


def write_export(path: Path, refused: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "goldenSetSize": 1,
                "privateMetadata": "do not upload",
                "questions": [
                    {
                        "id": "private-reference",
                        "answerable": True,
                        "refused": refused,
                        "citations": int(not refused),
                        "relevantCitations": int(not refused),
                        "privateText": "do not upload",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_report_publishes_aliases_and_detects_regressions(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "a.json", tmp_path / "b.json"
    case_map = tmp_path / "map.json"
    write_export(baseline, refused=False)
    write_export(candidate, refused=True)

    prepared = subprocess.run(
        [sys.executable, SCRIPT, "prepare", baseline, candidate, case_map],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "private-reference" not in prepared.stdout
    assert "do not upload" not in prepared.stdout
    assert json.loads(case_map.read_text()) == {"case-0001": "private-reference"}

    event, summary = tmp_path / "event.json", tmp_path / "summary.md"
    event.write_text(json.dumps({"inputs": json.loads(prepared.stdout)}))
    result = subprocess.run(
        [sys.executable, SCRIPT, "summarize-event"],
        env={
            **os.environ,
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["regressions"] == ["case-0001"]
    assert "Regressions: `case-0001`" in summary.read_text()


def test_workflow_refuses_unaliased_source_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_export(source, refused=False)
    event = tmp_path / "event.json"
    summary = tmp_path / "summary.md"
    payload = source.read_text()
    event.write_text(
        json.dumps({"inputs": dict.fromkeys(("baseline", "candidate"), payload)})
    )
    result = subprocess.run(
        [sys.executable, SCRIPT, "summarize-event"],
        env={
            **os.environ,
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "private-reference" not in result.stdout + result.stderr
    assert not summary.exists()
