import asyncio

from llm_eval.providers.base import AsyncChatModel


async def complete_many(
    model: AsyncChatModel, system: str, prompts: list[str], limit: int = 4
) -> list[str]:
    """Ask the model about every prompt, at most ``limit`` at a time.

    Results keep the order of ``prompts``.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}.")
    semaphore = asyncio.Semaphore(limit)

    async def limited(prompt: str) -> str:
        async with semaphore:
            return await model.complete(system, prompt)

    return await asyncio.gather(*(limited(prompt) for prompt in prompts))
