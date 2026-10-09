from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

Winner = Literal["A", "B", "tie"]

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    category: str
    split: Split


@dataclass(frozen=True)
class AppVersion:
    name: str
    system_prompt: str
    model: str
    temperature: float = 0.0


class JudgeVerdict(BaseModel):
    winner: Winner
    reason: str = Field(min_length=1)
