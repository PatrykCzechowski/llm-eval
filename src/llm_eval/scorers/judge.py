import asyncio

from pydantic import BaseModel, ValidationError

from llm_eval.models import JudgeGrade, JudgeVerdict, Sample, Winner
from llm_eval.providers.base import ChatModel
from llm_eval.scorers.base import Score

JUDGE_SYSTEM = "You are an impartial judge comparing two answers to a question."

JUDGE_PROMPT = """Question: {question}

Answer 1: {first}

Answer 2: {second}

Which answer is better? Judge helpfulness, correctness and clarity.
Do not let the order or length of the answers influence you.
Reply only with JSON: {{"winner": "A" | "B" | "tie", "reason": "..."}}
where A means Answer 1 and B means Answer 2."""

GRADE_SYSTEM = "You are a strict grader comparing an answer with a reference answer."

GRADE_PROMPT = """Question: {question}

Reference answer: {reference}

Answer to grade: {answer}

How well does the answer agree with the reference? Ignore style and length.
1 = wrong or contradicts the reference, 3 = partly correct, 5 = fully correct.
Reply only with JSON: {{"grade": 1 | 2 | 3 | 4 | 5, "reason": "..."}}"""

_SWAPPED: dict[Winner, Winner] = {"A": "B", "B": "A", "tie": "tie"}


class JudgeError(Exception):
    """The judge did not return a valid verdict."""


def _complete_validated[T: BaseModel](
    model: ChatModel, system: str, prompt: str, schema: type[T]
) -> T:
    """Ask the model and validate its JSON reply; retry once when it is invalid."""
    last_error: ValidationError | None = None
    for _ in range(2):
        raw = model.complete(system, prompt)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as error:
            last_error = error
    raise JudgeError(
        f"Judge returned an invalid {schema.__name__} twice."
    ) from last_error


def judge_once(model: ChatModel, question: str, first: str, second: str) -> Winner:
    """Judge one ordering of the answers; retry once on an invalid verdict.

    Raises JudgeError when both attempts are invalid.
    """
    prompt = JUDGE_PROMPT.format(question=question, first=first, second=second)
    return _complete_validated(model, JUDGE_SYSTEM, prompt, JudgeVerdict).winner


def judge_pair(model: ChatModel, question: str, answer_a: str, answer_b: str) -> Winner:
    """Judge both orderings; return the winner only when they agree, else tie."""
    original = judge_once(model, question, answer_a, answer_b)
    swapped = _SWAPPED[judge_once(model, question, answer_b, answer_a)]
    return original if original == swapped else "tie"


class JudgeScorer:
    """LLM judge grading agreement with the reference on a 1-5 scale.

    The grade is mapped linearly to 0.0-1.0. The synchronous model runs in a
    worker thread, so it does not block the event loop.
    """

    name = "judge"

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    async def score(self, sample: Sample, answer: str) -> Score:
        """Grade the answer; raise JudgeError after two invalid replies."""
        prompt = GRADE_PROMPT.format(
            question=sample.input, reference=sample.reference, answer=answer
        )
        grade = await asyncio.to_thread(
            _complete_validated, self.model, GRADE_SYSTEM, prompt, JudgeGrade
        )
        return Score(self.name, (grade.grade - 1) / 4, grade.reason)
