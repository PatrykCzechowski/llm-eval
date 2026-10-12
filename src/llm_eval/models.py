from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

Winner = Literal["A", "B", "tie"]

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Question:
    """MT-Bench question assigned to a data split."""

    id: str
    text: str
    category: str
    split: Split


@dataclass(frozen=True)
class Sample:
    """One dataset item: the input sent to the model and the expected answer."""

    id: str
    input: str
    reference: str


@dataclass(frozen=True)
class AppVersion:
    """Variant of the evaluated application: prompt, model and temperature."""

    name: str
    system_prompt: str
    model: str
    temperature: float = 0.0


class JudgeVerdict(BaseModel):
    """Structured verdict returned by the LLM judge."""

    winner: Winner
    reason: str = Field(min_length=1)


class JudgeGrade(BaseModel):
    """Structured 1-5 grade of how well an answer matches the reference."""

    grade: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1)
