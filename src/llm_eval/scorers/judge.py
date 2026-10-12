from pydantic import ValidationError

from llm_eval.models import JudgeVerdict, Winner
from llm_eval.providers.base import ChatModel

JUDGE_SYSTEM = "You are an impartial judge comparing two answers to a question."

JUDGE_PROMPT = """Question: {question}

Answer 1: {first}

Answer 2: {second}

Which answer is better? Judge helpfulness, correctness and clarity.
Do not let the order or length of the answers influence you.
Reply only with JSON: {{"winner": "A" | "B" | "tie", "reason": "..."}}
where A means Answer 1 and B means Answer 2."""

_SWAPPED: dict[Winner, Winner] = {"A": "B", "B": "A", "tie": "tie"}


class JudgeError(Exception):
    pass


def judge_once(model: ChatModel, question: str, first: str, second: str) -> Winner:
    prompt = JUDGE_PROMPT.format(question=question, first=first, second=second)
    last_error: ValidationError | None = None
    for _ in range(2):
        raw = model.complete(JUDGE_SYSTEM, prompt)
        try:
            return JudgeVerdict.model_validate_json(raw).winner
        except ValidationError as error:
            last_error = error
    raise JudgeError("Judge returned an invalid verdict twice.") from last_error


def judge_pair(model: ChatModel, question: str, answer_a: str, answer_b: str) -> Winner:
    original = judge_once(model, question, answer_a, answer_b)
    swapped = _SWAPPED[judge_once(model, question, answer_b, answer_a)]
    return original if original == swapped else "tie"
