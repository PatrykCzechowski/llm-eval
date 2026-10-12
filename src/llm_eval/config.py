"""Eval configuration loaded from YAML and validated with Pydantic."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelSpec(BaseModel):
    """Model under evaluation, e.g. ``ollama:llama3.2:3b`` with sampling params."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)


class RunnerSpec(BaseModel):
    """How the runner calls models: concurrency limit and retries."""

    model_config = ConfigDict(extra="forbid")

    concurrency: int = Field(default=4, gt=0)
    max_retries: int = Field(default=2, ge=0)


class GateSpec(BaseModel):
    """Regression gate: the run fails when ``metric`` drops below ``min``."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    min: float = Field(ge=0, le=1)


class EvalConfig(BaseModel):
    """Full description of one eval. Paths are relative to the repository root."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dataset: Path
    prompt: Path
    models: list[ModelSpec] = Field(min_length=1)
    scorers: list[str] = Field(min_length=1)
    runner: RunnerSpec = Field(default_factory=RunnerSpec)
    gate: GateSpec | None = None


def load_config(path: Path) -> EvalConfig:
    """Read a YAML eval config and validate it.

    Raises pydantic.ValidationError when the content does not match EvalConfig.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvalConfig.model_validate(data)
