"""Base model provider interface."""

from __future__ import annotations

from abc import ABC


class ModelProvider(ABC):  # noqa: B024
    """Abstract base class for model providers."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

    def run(self, prompt: str) -> str:
        """Execute the model with the given prompt and return the response.

        This method is optional - only CLI-based providers (ClaudeCode, OpenAICodex)
        need to implement it. LiteLLM-based providers use async_run() instead.

        Args:
        ----
            prompt: The prompt text to send to the model

        Returns:
        -------
            The model's response text

        Raises:
        ------
            RuntimeError: If the model execution fails
            NotImplementedError: If the provider doesn't support synchronous execution

        """
        msg = f"{self.__class__.__name__} does not support synchronous execution"
        raise NotImplementedError(msg)

    def get_provider_info(self) -> dict[str, str | float]:
        """Get provider metadata for output file embedding.

        Returns
        -------
            Dictionary with provider name, model name, and temperature

        """
        return {
            "provider": self.__class__.__name__.replace("Provider", "").lower(),
            "model": self.model_name or "unknown",
            "temperature": getattr(self, "temperature", 0.0),
        }
