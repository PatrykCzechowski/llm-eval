from typing import Protocol


class ChatModel(Protocol):
    """Synchronous chat model: one system and one user message in, text out."""

    def complete(self, system: str, user: str) -> str: ...


class AsyncChatModel(Protocol):
    """Asynchronous counterpart of ChatModel."""

    async def complete(self, system: str, user: str) -> str: ...
