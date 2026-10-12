import json
from pathlib import Path

import pytest

from llm_eval.data.loaders import load_dataset, read_jsonl
from llm_eval.models import Sample

REPO_ROOT = Path(__file__).parents[1]


def write_jsonl(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sample_line(sample_id: str) -> str:
    return json.dumps(
        {"id": sample_id, "input": f"question {sample_id}", "reference": "answer"}
    )


def test_read_jsonl_yields_records_with_line_numbers(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, ['{"a": 1}', "", '{"a": 2}'])

    assert list(read_jsonl(path)) == [(1, {"a": 1}), (3, {"a": 2})]


def test_read_jsonl_is_lazy(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, ['{"a": 1}', "not json"])
    records = read_jsonl(path)

    assert next(records) == (1, {"a": 1})
    with pytest.raises(ValueError):
        next(records)


def test_load_dataset_returns_samples(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [sample_line("q1"), sample_line("q2")])

    samples = load_dataset(path)

    assert samples == [
        Sample(id="q1", input="question q1", reference="answer"),
        Sample(id="q2", input="question q2", reference="answer"),
    ]


def test_load_dataset_reports_line_of_invalid_json(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [sample_line("q1"), sample_line("q2"), "{broken"])

    with pytest.raises(ValueError, match=r":3: invalid JSON"):
        load_dataset(path)


def test_load_dataset_reports_line_of_missing_field(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path, [sample_line("q1"), sample_line("q2"), '{"id": "q3", "input": "x"}']
    )

    with pytest.raises(ValueError, match=r":3: invalid sample \(fields: reference\)"):
        load_dataset(path)


def test_load_dataset_rejects_non_string_field(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, ['{"id": 1, "input": "x", "reference": "y"}'])

    with pytest.raises(ValueError, match=r":1: invalid sample \(fields: id\)"):
        load_dataset(path)


def test_load_dataset_rejects_non_object_line(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, ['["not", "an", "object"]'])

    with pytest.raises(ValueError, match=r":1: expected a JSON object"):
        load_dataset(path)


def test_load_dataset_rejects_duplicate_id(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [sample_line("q1"), sample_line("q1")])

    with pytest.raises(ValueError, match=r":2: duplicate id 'q1'"):
        load_dataset(path)


def test_repository_dataset_has_20_samples() -> None:
    samples = load_dataset(REPO_ROOT / "datasets" / "geografia.jsonl")

    assert len(samples) == 20
