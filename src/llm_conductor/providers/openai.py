"""OpenAI API provider with LiteLLM."""

from __future__ import annotations

import os
from typing import override

from llm_conductor.litellm_base import LiteLLMProvider


class OpenAIProvider(LiteLLMProvider):
    """Provider for OpenAI API (supports local models via OPENAI_BASE_URL)."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name)
        self.api_key = os.getenv("OPENAI_API_KEY", "dummy")
        self.base_url = os.getenv("OPENAI_BASE_URL")

        if not self.model_name:
            self.model_name = os.getenv("MODEL_NAME", "gpt-4")

    @override
    def _get_api_key(self) -> str:
        return self.api_key

    @override
    def _get_base_url(self) -> str | None:
        return self.base_url

    @override
    def _get_default_model(self) -> str:
        return "gpt-4"
