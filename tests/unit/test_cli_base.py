"""Unit tests for CLI base class."""

from __future__ import annotations

import pytest

from llm_conductor.cli_base import CLIProvider


class MockCLIProvider(CLIProvider):
    """Mock CLI provider for testing."""

    def __init__(self, model_name: str = "test-model", encoding: str = "cl100k_base"):
        self.model_name = model_name
        self._encoding = encoding
        self._run_called = False
        self._response = "Test response from CLI"

    def _get_encoding_name(self) -> str:
        """Return test encoding name."""
        return self._encoding

    def run(self, prompt: str) -> str:
        """Mock run method."""
        self._run_called = True
        return self._response

    def get_provider_info(self) -> dict[str, str | float]:
        """Return provider metadata."""
        return {
            "provider": "mock-cli",
            "model": self.model_name,
            "temperature": 0.0,
        }


class TestCLIProvider:
    """Test CLIProvider base class functionality."""

    def test_count_tokens_basic(self):
        """Test basic token counting."""
        provider = MockCLIProvider()
        text = "Hello, world!"
        tokens = provider.count_tokens(text)

        # Should return a positive integer
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_tokens_empty_string(self):
        """Test token counting with empty string."""
        provider = MockCLIProvider()
        tokens = provider.count_tokens("")

        # Empty string should have 0 tokens
        assert tokens == 0

    def test_count_tokens_longer_text(self):
        """Test that longer text has more tokens."""
        provider = MockCLIProvider()
        short_text = "Hello"
        long_text = (
            "Hello, this is a much longer piece of text that should have more tokens"
        )

        short_tokens = provider.count_tokens(short_text)
        long_tokens = provider.count_tokens(long_text)

        assert long_tokens > short_tokens

    def test_count_tokens_different_encodings(self):
        """Test token counting with different encodings."""
        provider_cl100k = MockCLIProvider(encoding="cl100k_base")
        provider_o200k = MockCLIProvider(encoding="o200k_base")

        text = "Hello, world! This is a test."

        # Both should return positive counts (exact values may differ)
        tokens_cl100k = provider_cl100k.count_tokens(text)
        tokens_o200k = provider_o200k.count_tokens(text)

        assert tokens_cl100k > 0
        assert tokens_o200k > 0

    def test_run_with_usage_returns_tuple(self):
        """Test that run_with_usage returns correct tuple format."""
        provider = MockCLIProvider()
        content, usage = provider.run_with_usage("Test prompt")

        assert isinstance(content, str)
        assert isinstance(usage, dict)

    def test_run_with_usage_calls_run(self):
        """Test that run_with_usage calls the run method."""
        provider = MockCLIProvider()
        provider.run_with_usage("Test prompt")

        assert provider._run_called is True

    def test_run_with_usage_returns_response(self):
        """Test that run_with_usage returns the provider's response."""
        provider = MockCLIProvider()
        provider._response = "Custom test response"

        content, usage = provider.run_with_usage("Test prompt")
        assert content == "Custom test response"

    def test_run_with_usage_includes_token_counts(self):
        """Test that usage dict includes token counts."""
        provider = MockCLIProvider()
        content, usage = provider.run_with_usage("Test prompt")

        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)

        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] > 0

    def test_run_with_usage_total_tokens_correct(self):
        """Test that total tokens equals sum of prompt and completion."""
        provider = MockCLIProvider()
        content, usage = provider.run_with_usage("Test prompt")

        expected_total = usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["total_tokens"] == expected_total

    def test_run_with_usage_includes_cost(self):
        """Test that usage dict includes cost calculation."""
        provider = MockCLIProvider()
        content, usage = provider.run_with_usage("Test prompt")

        assert "estimated_cost_usd" in usage
        assert isinstance(usage["estimated_cost_usd"], int | float)
        assert usage["estimated_cost_usd"] >= 0

    def test_run_with_usage_includes_cost_method(self):
        """Test that usage dict includes cost calculation method."""
        provider = MockCLIProvider()
        content, usage = provider.run_with_usage("Test prompt")

        assert "cost_calculation_method" in usage
        assert usage["cost_calculation_method"] == "estimated_tiktoken"

    def test_run_with_usage_cost_calculation(self):
        """Test that cost is calculated based on tokens and model."""
        provider = MockCLIProvider(model_name="gpt-4o-mini")
        content, usage = provider.run_with_usage("Test prompt with some content")

        # gpt-4o-mini is cheap, cost should be very small
        assert usage["estimated_cost_usd"] < 0.01  # Less than a cent
        assert usage["estimated_cost_usd"] > 0  # But greater than zero

    def test_run_with_usage_different_prompts_different_costs(self):
        """Test that different prompt sizes result in different costs."""
        provider = MockCLIProvider()

        short_prompt = "Hi"
        long_prompt = "This is a much longer prompt with many more words and tokens that should result in a higher cost calculation"

        _, usage_short = provider.run_with_usage(short_prompt)
        _, usage_long = provider.run_with_usage(long_prompt)

        # Longer prompt should have higher cost
        assert usage_long["prompt_tokens"] > usage_short["prompt_tokens"]
        assert usage_long["estimated_cost_usd"] > usage_short["estimated_cost_usd"]

    def test_abstract_get_encoding_name(self):
        """Test that _get_encoding_name is abstract."""

        class IncompleteCLIProvider(CLIProvider):
            def run(self, prompt: str) -> str:
                return "test"

            def get_provider_info(self) -> dict[str, str | float]:
                return {}

        # Should not be able to instantiate without implementing _get_encoding_name
        with pytest.raises(TypeError, match="abstract"):
            IncompleteCLIProvider()

    def test_abstract_run(self):
        """Test that run is abstract (inherited from ModelProvider)."""

        # This test verifies the abstract method requirement
        # by checking that CLIProvider cannot be instantiated directly
        with pytest.raises(TypeError):
            # Should not be able to instantiate CLIProvider directly
            CLIProvider()  # type: ignore

    def test_model_name_used_for_pricing(self):
        """Test that model_name is used for cost calculation."""
        # Use a known expensive model
        provider_expensive = MockCLIProvider(model_name="gpt-4")
        provider_cheap = MockCLIProvider(model_name="gpt-4o-mini")

        # Same prompt for both
        prompt = "Calculate this"

        _, usage_expensive = provider_expensive.run_with_usage(prompt)
        _, usage_cheap = provider_cheap.run_with_usage(prompt)

        # Expensive model should cost more
        assert usage_expensive["estimated_cost_usd"] > usage_cheap["estimated_cost_usd"]

    def test_count_tokens_unicode(self):
        """Test token counting with unicode characters."""
        provider = MockCLIProvider()

        # Unicode characters
        text_unicode = "Hello 👋 世界 🌍"
        tokens = provider.count_tokens(text_unicode)

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_newlines(self):
        """Test token counting with newlines."""
        provider = MockCLIProvider()

        text_multiline = "Line 1\nLine 2\nLine 3"
        tokens = provider.count_tokens(text_multiline)

        assert tokens > 0

    def test_run_with_usage_empty_response(self):
        """Test run_with_usage handles empty response."""
        provider = MockCLIProvider()
        provider._response = ""

        content, usage = provider.run_with_usage("Test prompt")

        assert content == ""
        assert usage["completion_tokens"] == 0
        # Prompt tokens should still be counted
        assert usage["prompt_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"]

    def test_run_with_usage_large_response(self):
        """Test run_with_usage with large response."""
        provider = MockCLIProvider()
        # Generate a large response
        provider._response = "word " * 1000  # ~1000 tokens

        content, usage = provider.run_with_usage("Test")

        # Should have many completion tokens
        assert usage["completion_tokens"] > 500
        assert usage["total_tokens"] > 500
