"""Ollama local server provider with LiteLLM."""

from __future__ import annotations

from typing import override

from llm_conductor.litellm_base import LiteLLMProvider


class OllamaProvider(LiteLLMProvider):
    """Provider for Ollama local server."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name)
        # Ollama default endpoint
        self.base_url = "http://localhost:11434"
        self.api_key = "ollama"  # LiteLLM requires a non-empty key for Ollama

        if not self.model_name:
            self.model_name = "ollama/gpt-oss:120b"

    @override
    def _get_api_key(self) -> str:
        return self.api_key

    @override
    def _get_base_url(self) -> str | None:
        return self.base_url

    @override
    def _get_default_model(self) -> str:
        return "ollama/gpt-oss:120b"
