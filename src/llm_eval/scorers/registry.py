"""Registry that turns scorer names from the YAML config into scorer objects."""

from collections.abc import Callable

from llm_eval.providers.base import ChatModel
from llm_eval.scorers.base import Scorer
from llm_eval.scorers.exact import ExactMatch, RegexMatch
from llm_eval.scorers.judge import JudgeScorer

SCORERS: dict[str, Callable[[], Scorer]] = {
    "exact": ExactMatch,
    "regex": RegexMatch,
}
JUDGE = "judge"


def available_scorers() -> list[str]:
    """Sorted names accepted by build_scorers."""
    return sorted([*SCORERS, JUDGE])


def build_scorers(
    names: list[str], judge_model: ChatModel | None = None
) -> list[Scorer]:
    """Create scorers by name; ``judge`` additionally needs ``judge_model``.

    Raises ValueError for an unknown name or a missing judge model.
    """
    scorers: list[Scorer] = []
    for name in names:
        if name == JUDGE:
            if judge_model is None:
                raise ValueError("Scorer 'judge' needs a judge model.")
            scorers.append(JudgeScorer(judge_model))
        elif name in SCORERS:
            scorers.append(SCORERS[name]())
        else:
            raise ValueError(
                f"Unknown scorer {name!r}. Available: {', '.join(available_scorers())}."
            )
    return scorers
