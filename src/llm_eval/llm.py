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
