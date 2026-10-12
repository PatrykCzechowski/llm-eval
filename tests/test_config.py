from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_eval.config import EvalConfig, load_config

REPO_ROOT = Path(__file__).parents[1]

VALID_CONFIG = """\
name: geo
dataset: datasets/geo.jsonl
prompt: prompts/qa.txt
models:
  - id: ollama:llama3.2:3b
    params: { temperature: 0 }
scorers: [exact]
runner: { concurrency: 2, max_retries: 1 }
gate: { metric: exact, min: 0.85 }
"""


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_parses_valid_file(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG))

    assert config.name == "geo"
    assert config.dataset == Path("datasets/geo.jsonl")
    assert config.models[0].id == "ollama:llama3.2:3b"
    assert config.models[0].params == {"temperature": 0}
    assert config.runner.concurrency == 2
    assert config.gate is not None
    assert config.gate.min == 0.85


def test_load_config_uses_defaults_for_optional_sections(tmp_path: Path) -> None:
    content = VALID_CONFIG.split("runner:")[0]

    config = load_config(write_config(tmp_path, content))

    assert config.runner.concurrency == 4
    assert config.runner.max_retries == 2
    assert config.gate is None


def test_load_config_rejects_empty_models(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace(
        "models:\n  - id: ollama:llama3.2:3b\n    params: { temperature: 0 }",
        "models: []",
    )

    with pytest.raises(ValidationError, match="models"):
        load_config(write_config(tmp_path, content))


def test_load_config_rejects_missing_models(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace(
        "models:\n  - id: ollama:llama3.2:3b\n    params: { temperature: 0 }\n", ""
    )

    with pytest.raises(ValidationError, match="models"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("concurrency", [0, -1])
def test_load_config_rejects_non_positive_concurrency(
    tmp_path: Path, concurrency: int
) -> None:
    content = VALID_CONFIG.replace("concurrency: 2", f"concurrency: {concurrency}")

    with pytest.raises(ValidationError, match="concurrency"):
        load_config(write_config(tmp_path, content))


def test_load_config_rejects_negative_max_retries(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("max_retries: 1", "max_retries: -1")

    with pytest.raises(ValidationError, match="max_retries"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("gate_min", [-0.1, 1.5])
def test_load_config_rejects_gate_min_outside_unit_range(
    tmp_path: Path, gate_min: float
) -> None:
    content = VALID_CONFIG.replace("min: 0.85", f"min: {gate_min}")

    with pytest.raises(ValidationError, match="min"):
        load_config(write_config(tmp_path, content))


def test_load_config_rejects_unknown_field(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("runner:", "runer:")

    with pytest.raises(ValidationError, match="runer"):
        load_config(write_config(tmp_path, content))


def test_repository_config_is_valid() -> None:
    config = load_config(REPO_ROOT / "configs" / "geografia.yaml")

    assert isinstance(config, EvalConfig)
    assert (REPO_ROOT / config.dataset).exists()
    assert (REPO_ROOT / config.prompt).exists()
