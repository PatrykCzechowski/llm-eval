"""Ask a local Ollama model one question and print the answer.

Configure with environment variables, e.g. to use Ollama on another machine:
    OLLAMA_HOST=http://192.168.1.50:11434 uv run python scripts/try_ollama.py
"""

import os

from llm_eval.llm import DEFAULT_OLLAMA_HOST, OllamaModel


def main() -> None:
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    model = OllamaModel(model=model_name, host=host)

    answer = model.complete(
        system="You are a concise assistant. Answer in one sentence.",
        user="What is the capital of Poland?",
    )
    print(f"[{model_name} @ {host}] {answer}")


if __name__ == "__main__":
    main()
