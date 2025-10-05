"""Unit tests for model providers (runner, litellm_base, and provider implementations)."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from llm_conductor.base import ModelProvider

# Configure pytest-anyio to only use asyncio backend
pytestmark = pytest.mark.anyio(backends=["asyncio"])


class FakeStreamProvider:
    """Test provider that returns canned responses without external API calls."""

    def __init__(
        self, responses: list[tuple[str, str | None]], model_name: str = "test-model"
    ) -> None:
        """Initialize fake provider with canned responses.

        Args:
        ----
            responses: List of (content, finish_reason) tuples to return sequentially
            model_name: Model name for the provider

        """
        from llm_conductor.litellm_base import LiteLLMProvider

        # Store responses before calling super().__init__
        self._responses = responses
        self._call_count = 0

        # Create a temporary class that inherits from LiteLLMProvider
        class _FakeProvider(LiteLLMProvider):
            def _get_api_key(self) -> str:
                return "fake-key"

            def _get_base_url(self) -> str | None:
                return None

            def _get_default_model(self) -> str:
                return model_name

        self._provider = _FakeProvider(model_name=model_name)

        # Override async_run to return canned responses

        async def fake_async_run(
            _prompt: str, _stream_to_stdout: bool = False
        ) -> tuple[str, dict | None, str | None]:
            if self._call_count >= len(self._responses):
                msg = "No more test responses available"
                raise RuntimeError(msg)

            content, finish_reason = self._responses[self._call_count]
            self._call_count += 1

            # Simulate continuation: if finish_reason is "length", accumulate next response
            full_content = content
            while finish_reason == "length" and self._call_count < len(self._responses):
                next_content, finish_reason = self._responses[self._call_count]
                self._call_count += 1
                full_content += next_content

            return (full_content, None, None)

        self._provider.async_run = fake_async_run

    async def async_run(
        self, prompt: str, stream_to_stdout: bool = False
    ) -> tuple[str, dict | None, str | None]:
        """Delegate to provider's async_run."""
        return await self._provider.async_run(prompt, stream_to_stdout)

    @property
    def call_count(self) -> int:
        """Return number of API calls made."""
        return self._call_count

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._provider.model_name


class TestProviderResponses:
    """Test provider response handling with FakeStreamProvider."""

    async def test_simple_response(self):
        """Test basic single response."""
        provider = FakeStreamProvider([("Test content", "stop")])
        result, _usage, _gen_id = await provider.async_run("test prompt")
        assert result == "Test content"
        assert provider.call_count == 1

    async def test_continuation_on_length_finish_reason(self):
        """Test that continuation happens when finish_reason == 'length'."""
        provider = FakeStreamProvider(
            [("First part", "length"), (" second part", "stop")]
        )
        result, _usage, _gen_id = await provider.async_run("test prompt")
        assert result == "First part second part"
        assert provider.call_count == 2

    async def test_multiple_continuations(self):
        """Test multiple continuations in sequence."""
        provider = FakeStreamProvider(
            [("Part 1", "length"), (" Part 2", "length"), (" Part 3", "stop")]
        )
        result, _usage, _gen_id = await provider.async_run("test prompt")
        assert result == "Part 1 Part 2 Part 3"
        assert provider.call_count == 3

    async def test_exhausted_responses_raises_error(self):
        """Test that exhausting responses raises RuntimeError."""
        provider = FakeStreamProvider([("First", "stop")])
        # First call succeeds
        result, _, _ = await provider.async_run("test prompt")
        assert result == "First"
        # Second call should raise because responses are exhausted
        with pytest.raises(RuntimeError, match="No more test responses"):
            await provider.async_run("test prompt")


class StubSyncProvider:
    """Stub sync provider for testing with call tracking."""

    def __init__(self, response: str = "Test response") -> None:
        self.model_name = "test-model"
        self.__class__.__name__ = "StubSyncProvider"
        self._response = response
        self.call_count = 0

    def run(self, prompt: str) -> str:
        """Return canned response and track calls."""
        self.call_count += 1
        return self._response

    def get_provider_info(self) -> dict[str, str | float]:
        """Return provider metadata."""
        return {
            "provider": "stub-sync",
            "model": self.model_name,
            "temperature": 0.0,
        }


class StubAsyncProvider:
    """Stub async provider for testing with call tracking."""

    def __init__(self, response: str = "Test async response") -> None:
        self.model_name = "test-model"
        self.__class__.__name__ = "StubAsyncProvider"
        self._response = response
        self.call_count = 0

    async def async_run(
        self, prompt: str, stream_to_stdout: bool = False
    ) -> tuple[str, dict | None, str | None]:
        """Return canned response as tuple and track calls."""
        self.call_count += 1
        return (self._response, None, None)

    def get_provider_info(self) -> dict[str, str | float]:
        """Return provider metadata."""
        return {
            "provider": "stub-async",
            "model": self.model_name,
            "temperature": 0.0,
        }


class TestRunner:
    """Test runner.py batch and single mode execution."""

    @pytest.fixture
    def mock_provider(self) -> ModelProvider:
        """Create a stub sync provider for testing."""
        return StubSyncProvider()

    @pytest.fixture
    def mock_async_provider(self) -> ModelProvider:
        """Create a stub async provider for testing."""
        return StubAsyncProvider()

    def test_sync_process_single_task(self, mock_provider, tmp_path: Path):
        """Test process_single_task with sync provider."""
        from llm_conductor.runner import process_single_task

        # Create test prompt file
        prompt_file = tmp_path / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        # Pass base path without extension (extension determined by content)
        output_base = tmp_path / "output"
        test_hash = "abc123"

        output_name, success, error, elapsed, _usage, _generation_id = (
            process_single_task(
                mock_provider, str(prompt_file), str(output_base), test_hash
            )
        )

        assert success is True
        assert error is None
        assert output_name == "output.md"
        assert elapsed > 0

        # Check that markdown file was created (content detection)
        output_file = tmp_path / "output.md"
        assert output_file.exists()

        # Check hash was embedded
        content = output_file.read_text()
        assert test_hash in content
        assert "<!-- Prompt Hash:" in content

    async def test_async_process_single_task(self, mock_async_provider, tmp_path: Path):
        """Test async_process_single_task with async provider."""
        from llm_conductor.runner import async_process_single_task

        prompt_file = tmp_path / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        # Pass base path without extension (extension determined by content)
        output_base = tmp_path / "output"
        test_hash = "def456"

        (
            output_name,
            success,
            error,
            _elapsed,
            _usage,
            _generation_id,
        ) = await async_process_single_task(
            mock_async_provider, str(prompt_file), str(output_base), test_hash
        )

        assert success is True
        assert error is None

        # Check that markdown file was created (content detection)
        output_file = tmp_path / "output.md"
        assert output_name == "output.md"
        assert output_file.exists()

        content = output_file.read_text()
        assert test_hash in content

    def test_hash_embedding_json_dict(self, tmp_path: Path):
        """Test hash embedding in JSON dict output."""
        from llm_conductor.runner import process_single_task

        # Create provider with JSON response
        provider = StubSyncProvider(response='{"key": "value"}')

        prompt_file = tmp_path / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        # Pass base path without extension (extension determined by content)
        output_base = tmp_path / "output"
        test_hash = "json_dict_hash"

        process_single_task(provider, str(prompt_file), str(output_base), test_hash)

        # Check that JSON file was created (content detection)
        output_file = tmp_path / "output.json"
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert data["_prompt_hash"] == test_hash
        assert data["key"] == "value"

    def test_hash_embedding_json_array(self, tmp_path: Path):
        """Test hash embedding with JSON array (should write as-is)."""
        from llm_conductor.runner import process_single_task

        # Create provider with JSON array response
        provider = StubSyncProvider(response='[{"item": 1}, {"item": 2}]')

        prompt_file = tmp_path / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        # Pass base path without extension (extension determined by content)
        output_base = tmp_path / "output"

        process_single_task(provider, str(prompt_file), str(output_base), "hash")

        # Check that JSON file was created (content detection)
        output_file = tmp_path / "output.json"
        assert output_file.exists()

        # Array should be written as-is (no hash injection)
        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_change_detection_skip_when_hash_matches(
        self, mock_provider, tmp_path: Path, monkeypatch
    ):
        """Test that tasks are skipped when hash matches existing output."""
        from llm_conductor.runner import (
            compute_prompt_hash,
            extract_existing_hash,
            run_batch_mode,
        )

        # Create prompt file
        prompt_file = tmp_path / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        # Compute the actual hash of the prompt
        test_hash = compute_prompt_hash(prompt_file)

        # Create existing output with matching hash (output derives from input: test.prompt.md → test.md)
        output_file = tmp_path / "test.md"
        output_file.write_text(f"<!-- Prompt Hash: {test_hash} -->\n\nExisting content")

        # Verify hash extraction works
        extracted = extract_existing_hash(str(output_file))
        assert extracted == test_hash

        # Build prompt files list
        prompt_files = list(tmp_path.glob("*.prompt.md"))

        # Run batch mode (should skip because hash matches)
        exit_code = run_batch_mode(
            mock_provider,
            prompt_files,
            parallelism=1,
            verbose=False,
            force=False,
            input_dir=tmp_path,
            output_dir=tmp_path,
        )

        # Provider should NOT be called since hash matches
        assert mock_provider.call_count == 0
        assert exit_code == 0

        # Verify output file still exists and unchanged
        assert output_file.exists()
        assert "Existing content" in output_file.read_text()

    def test_change_detection_regenerate_when_hash_differs(
        self, mock_provider, tmp_path: Path, monkeypatch
    ):
        """Test that tasks run when hash differs from existing output."""
        from llm_conductor.runner import run_batch_mode

        # Create prompt file
        prompt_file = tmp_path / "test.prompt.md"
        prompt_file.write_text("Updated prompt")

        # Create existing output with old hash (output derives from input: test.prompt.md → test.md)
        output_file = tmp_path / "test.md"
        output_file.write_text("<!-- Prompt Hash: old_hash -->\n\nOld content")

        # Build prompt files list
        prompt_files = list(tmp_path.glob("*.prompt.md"))

        run_batch_mode(
            mock_provider,
            prompt_files,
            parallelism=1,
            verbose=False,
            force=False,
            input_dir=tmp_path,
            output_dir=tmp_path,
        )

        # Provider SHOULD be called since hash changed
        assert mock_provider.call_count == 1

    def test_force_mode_regenerates_all(
        self, mock_provider, tmp_path: Path, monkeypatch
    ):
        """Test that force=True regenerates even when hash matches."""
        from llm_conductor.runner import compute_prompt_hash, run_batch_mode

        # Create prompt file
        prompt_file = tmp_path / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        # Compute the actual hash of the prompt
        test_hash = compute_prompt_hash(prompt_file)

        # Create existing output with matching hash (output derives from input: test.prompt.md → test.md)
        output_file = tmp_path / "test.md"
        output_file.write_text(f"<!-- Prompt Hash: {test_hash} -->\n\nExisting")

        # Build prompt files list
        prompt_files = list(tmp_path.glob("*.prompt.md"))

        run_batch_mode(
            mock_provider,
            prompt_files,
            parallelism=1,
            verbose=False,
            force=True,  # Force regeneration
            input_dir=tmp_path,
            output_dir=tmp_path,
        )

        # Should be called even though hash matches
        assert mock_provider.call_count == 1

    def test_batch_mode_async_concurrency(
        self, mock_async_provider, tmp_path: Path, monkeypatch
    ):
        """Test that async batch mode respects parallelism limit."""
        from llm_conductor.runner import run_batch_mode

        # Create 10 tasks
        for i in range(10):
            prompt_file = tmp_path / f"prompt_{i}.prompt.md"
            prompt_file.write_text(f"Prompt {i}")

        # Track concurrency
        max_concurrent = 0
        current_concurrent = 0

        original_async_run = mock_async_provider.async_run

        async def tracked_async_run(prompt: str, **kwargs: Any):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)  # Simulate work
            result = await original_async_run(prompt, **kwargs)
            current_concurrent -= 1
            return result

        mock_async_provider.async_run = tracked_async_run

        # Build prompt files list
        prompt_files = list(tmp_path.glob("*.prompt.md"))

        # run_batch_mode uses asyncio.run internally, so we can call it synchronously
        run_batch_mode(
            mock_async_provider,
            prompt_files,
            parallelism=3,  # Limit to 3 concurrent
            verbose=False,
            force=False,
            input_dir=tmp_path,
            output_dir=tmp_path,
        )

        # Should never exceed parallelism limit
        assert max_concurrent <= 3


class TestProviderFactory:
    """Test provider factory and provider initialization."""

    def test_get_provider_openai(self, monkeypatch):
        """Test get_provider returns OpenAI provider."""
        from llm_conductor import get_provider

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = get_provider("openai", model_name="gpt-4", reasoning="default")
        assert provider.__class__.__name__ == "OpenAIProvider"
        assert provider.model_name == "gpt-4"

    def test_get_provider_openrouter(self, monkeypatch):
        """Test get_provider returns OpenRouter provider."""
        from llm_conductor import get_provider

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        provider = get_provider(
            "openrouter", model_name="anthropic/claude-3.5-sonnet", reasoning="default"
        )
        assert provider.__class__.__name__ == "OpenRouterProvider"

    def test_get_provider_invalid_raises_error(self):
        """Test get_provider raises error for invalid provider."""
        from llm_conductor import get_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid_provider", model_name=None, reasoning="default")

    def test_model_name_preparation_with_prefix(self, monkeypatch):
        """Test that model names are prefixed correctly for OpenRouter."""
        from llm_conductor.providers.openrouter import OpenRouterProvider

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        provider = OpenRouterProvider(model_name="anthropic/claude-3.5-sonnet")
        # Test that _prepare_model_name applies the prefix correctly
        prepared_name = provider._prepare_model_name(provider.model_name)
        assert prepared_name == "openrouter/anthropic/claude-3.5-sonnet"
        # Verify stored model_name doesn't have prefix (prefix applied at runtime)
        assert provider.model_name == "anthropic/claude-3.5-sonnet"
