"""Dataset loading from JSONL files."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from llm_eval.models import Sample

_SAMPLE_ADAPTER = TypeAdapter(Sample)


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` for each non-empty line, one at a time.

    Raises ValueError with the line number when a line is not a JSON object.
    """
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                # Bad file content, not a caller passing a wrong type.
                raise ValueError(f"{path}:{line_number}: expected a JSON object")  # noqa: TRY004
            yield line_number, record


def load_dataset(path: Path) -> list[Sample]:
    """Load samples with ``id``, ``input`` and ``reference`` string fields.

    Raises ValueError with the line number for invalid lines and duplicate ids.
    """
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    for line_number, record in read_jsonl(path):
        try:
            sample = _SAMPLE_ADAPTER.validate_python(record)
        except ValidationError as error:
            fields = ", ".join(str(e["loc"][0]) for e in error.errors() if e["loc"])
            raise ValueError(
                f"{path}:{line_number}: invalid sample (fields: {fields})"
            ) from error
        if sample.id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {sample.id!r}")
        seen_ids.add(sample.id)
        samples.append(sample)
    return samples
