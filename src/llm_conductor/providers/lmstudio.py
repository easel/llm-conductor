"""LM Studio local server provider with LiteLLM."""

from __future__ import annotations

from typing import override

from llm_conductor.litellm_base import LiteLLMProvider


class LMStudioProvider(LiteLLMProvider):
    """Provider for LM Studio local server (pre-configured endpoint)."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name)
        # Pre-configured LM Studio endpoint
        self.base_url = "http://192.168.2.106:1234/v1"
        self.api_key = "dummy"  # Not used by LM Studio

        if not self.model_name:
            self.model_name = "openai/gpt-oss-120b"

    @override
    def _get_api_key(self) -> str:
        return self.api_key

    @override
    def _get_base_url(self) -> str | None:
        return self.base_url

    @override
    def _get_default_model(self) -> str:
        return "openai/gpt-oss-120b"
