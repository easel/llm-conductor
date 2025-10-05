"""OpenRouter API provider with LiteLLM."""

from __future__ import annotations

import os
from typing import override

from llm_conductor.litellm_base import LiteLLMProvider


class OpenRouterProvider(LiteLLMProvider):
    """Provider for OpenRouter API (unified LLM routing service)."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Set OpenRouter app identification
        os.environ["OR_APP_NAME"] = "pulseflow"
        os.environ["OR_SITE_URL"] = os.getenv(
            "OPENROUTER_APP_URL", "https://github.com/synaptiq"
        )

        if not self.api_key:
            msg = "OPENROUTER_API_KEY environment variable is required"
            raise RuntimeError(msg)

        if not self.model_name:
            self.model_name = os.getenv("MODEL_NAME", "openai/gpt-5")

    @override
    def _get_api_key(self) -> str:
        return self.api_key  # type: ignore[return-value]

    @override
    def _get_base_url(self) -> str | None:
        return self.base_url

    @override
    def _get_default_model(self) -> str:
        return "openai/gpt-5"

    @override
    def _get_model_prefix(self) -> str:
        """OpenRouter requires 'openrouter/' prefix for model names."""
        return "openrouter/"
