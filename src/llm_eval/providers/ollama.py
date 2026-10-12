import ollama

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaModel:
    """ChatModel backed by a local or remote Ollama server."""

    def __init__(self, model: str, host: str = DEFAULT_OLLAMA_HOST) -> None:
        self.model = model
        self._client = ollama.Client(host=host)

    def complete(self, system: str, user: str) -> str:
        """Send one chat request; raise RuntimeError when the reply has no content."""
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
