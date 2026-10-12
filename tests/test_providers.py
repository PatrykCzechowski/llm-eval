import asyncio
from typing import Any

import ollama
import pytest

from llm_eval.core.runner import complete_many
from llm_eval.providers.base import AsyncChatModel, ChatModel
from llm_eval.providers.fake import FakeAsyncModel, FakeModel, ScriptedModel
from llm_eval.providers.ollama import OllamaModel


def test_fake_model_is_chat_model() -> None:
    model: ChatModel = FakeModel(answer="4")

    assert model.complete("Be brief.", "What is 2+2?") == "4"


def test_fake_model_records_calls() -> None:
    model = FakeModel(answer="4")

    model.complete("Be brief.", "What is 2+2?")

    assert model.calls == [("Be brief.", "What is 2+2?")]


def test_ollama_model_is_chat_model() -> None:
    model: ChatModel = OllamaModel(model="llama3.2:3b")

    assert isinstance(model, OllamaModel)


def test_ollama_model_sends_system_and_user_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    def fake_chat(self: ollama.Client, **kwargs: Any) -> ollama.ChatResponse:
        sent.update(kwargs)
        return ollama.ChatResponse(
            message=ollama.Message(role="assistant", content="4")
        )

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)
    model = OllamaModel(model="llama3.2:3b")

    answer = model.complete("Be brief.", "What is 2+2?")

    assert answer == "4"
    assert sent["model"] == "llama3.2:3b"
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is 2+2?"},
    ]


def test_ollama_model_raises_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(self: ollama.Client, **kwargs: Any) -> ollama.ChatResponse:
        return ollama.ChatResponse(message=ollama.Message(role="assistant"))

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)
    model = OllamaModel(model="llama3.2:3b")

    with pytest.raises(RuntimeError):
        model.complete("Be brief.", "What is 2+2?")


@pytest.mark.asyncio
async def test_fake_async_model_is_async_chat_model() -> None:
    model: AsyncChatModel = FakeAsyncModel()

    assert (
        await model.complete("Be brief.", "What is 2+2?") == "answer to: What is 2+2?"
    )


class ReverseFinishModel:
    """Answers later prompts faster, so completion order is reversed."""

    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts

    async def complete(self, system: str, user: str) -> str:
        remaining = len(self.prompts) - self.prompts.index(user)
        await asyncio.sleep(0.005 * remaining)
        return f"answer to: {user}"


@pytest.mark.asyncio
async def test_complete_many_keeps_prompt_order() -> None:
    prompts = [f"question {i}" for i in range(10)]
    model = ReverseFinishModel(prompts)

    answers = await complete_many(model, "Be brief.", prompts, limit=3)

    assert answers == [f"answer to: {prompt}" for prompt in prompts]


@pytest.mark.asyncio
async def test_complete_many_respects_concurrency_limit() -> None:
    model = FakeAsyncModel()
    prompts = [f"question {i}" for i in range(10)]

    await complete_many(model, "Be brief.", prompts, limit=3)

    assert model.max_active == 3


@pytest.mark.asyncio
async def test_complete_many_returns_empty_list_for_no_prompts() -> None:
    assert await complete_many(FakeAsyncModel(), "Be brief.", []) == []


@pytest.mark.asyncio
async def test_complete_many_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError):
        await complete_many(FakeAsyncModel(), "Be brief.", ["question"], limit=0)


def test_scripted_model_returns_answers_in_order() -> None:
    model: ChatModel = ScriptedModel(["first", "second"])

    assert model.complete("s", "u1") == "first"
    assert model.complete("s", "u2") == "second"


def test_scripted_model_raises_when_answers_run_out() -> None:
    model = ScriptedModel([])

    with pytest.raises(RuntimeError):
        model.complete("s", "u")
