from typing import Any

import ollama
import pytest

from llm_eval.llm import ChatModel, FakeModel, OllamaModel


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
