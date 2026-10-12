from typing import Protocol


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AsyncChatModel(Protocol):
    async def complete(self, system: str, user: str) -> str: ...
