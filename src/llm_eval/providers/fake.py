import asyncio


class FakeModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer


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


class ScriptedModel:
    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        if not self._answers:
            raise RuntimeError("ScriptedModel has no more answers.")
        self.calls.append((system, user))
        return self._answers.pop(0)
