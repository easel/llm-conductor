"""Base class for CLI-based model providers with token counting."""

from __future__ import annotations

from abc import abstractmethod

from llm_conductor.base import ModelProvider
from llm_conductor.pricing import calculate_cost


class CLIProvider(ModelProvider):
    """Abstract base class for CLI-based providers with tiktoken token counting.

    Subclasses must implement:
    - run(prompt: str) -> str: Execute the CLI command
    - _get_encoding_name() -> str: Return tiktoken encoding name
    """

    @abstractmethod
    def _get_encoding_name(self) -> str:
        """Get the tiktoken encoding name for this provider.

        Returns
        -------
            Encoding name (e.g., "cl100k_base", "o200k_base")

        """

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
        ----
            text: The text to count tokens for

        Returns:
        -------
            Number of tokens

        """
        try:
            import tiktoken
        except ImportError as e:
            msg = "tiktoken not installed. Install with: uv add tiktoken"
            raise RuntimeError(msg) from e

        encoding_name = self._get_encoding_name()
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))

    def run_with_usage(self, prompt: str) -> tuple[str, dict]:
        """Execute model with prompt and return content with token usage.

        Args:
        ----
            prompt: The prompt text to send to the model

        Returns:
        -------
            Tuple of (content, usage_dict) where usage_dict contains:
                - prompt_tokens: Number of tokens in prompt
                - completion_tokens: Number of tokens in response
                - total_tokens: Sum of prompt and completion tokens
                - estimated_cost_usd: Estimated cost in USD
                - cost_calculation_method: "estimated_tiktoken"

        """
        # Count prompt tokens
        prompt_tokens = self.count_tokens(prompt)

        # Run the model (implemented by subclass)
        content = self.run(prompt)

        # Count completion tokens
        completion_tokens = self.count_tokens(content)

        # Calculate estimated cost
        model_name = self.model_name or "default"
        estimated_cost = calculate_cost(prompt_tokens, completion_tokens, model_name)

        # Build usage dict
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": estimated_cost,
            "cost_calculation_method": "estimated_tiktoken",
        }

        return (content, usage)
