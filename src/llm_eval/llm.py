import asyncio
from typing import Protocol

import ollama

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class FakeModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer


class OllamaModel:
    def __init__(self, model: str, host: str = DEFAULT_OLLAMA_HOST) -> None:
        self.model = model
        self._client = ollama.Client(host=host)

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.message.content
        if content is None:
            raise RuntimeError(f"Model {self.model!r} returned no content.")
        return content


class AsyncChatModel(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class FakeAsyncModel:
    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0

    async def complete(self, system: str, user: str) -> str:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            return f"answer to: {user}"
        finally:
            self.active -= 1


async def complete_many(
    model: AsyncChatModel, system: str, prompts: list[str], limit: int = 4
) -> list[str]:
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}.")
    semaphore = asyncio.Semaphore(limit)

    async def limited(prompt: str) -> str:
        async with semaphore:
            return await model.complete(system, prompt)

    return await asyncio.gather(*(limited(prompt) for prompt in prompts))
