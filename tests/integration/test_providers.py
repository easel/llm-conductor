"""Integration tests for all LLM providers.

These tests verify that each provider can successfully process a trivial prompt.
Tests are automatically skipped if the provider is not available or configured.

Run with: pytest packages/llm-conductor/tests/integration/test_providers.py -v

Environment variables required (per provider):
- claude-code: Requires `claude` CLI installed
- openai-codex: Requires `codex` CLI installed
- openai: Requires OPENAI_API_KEY
- openrouter: Requires OPENROUTER_API_KEY
- lmstudio: Requires LM Studio server running at http://localhost:1234
- ollama: Requires Ollama server running at http://localhost:11434
"""

from __future__ import annotations

import asyncio
import os
import shutil

import pytest

# Trivial test prompt that should work with any model
TEST_PROMPT = """Please respond with a simple JSON object containing a greeting.

Example output:
{"message": "Hello, world!"}

Respond with only the JSON object, no additional text."""


def is_claude_code_available() -> bool:
    """Check if claude CLI is available."""
    return shutil.which("claude") is not None


def is_openai_codex_available() -> bool:
    """Check if codex CLI is available."""
    return shutil.which("codex") is not None


def is_openai_configured() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(os.getenv("OPENAI_API_KEY"))


def is_openrouter_configured() -> bool:
    """Check if OpenRouter API key is configured."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


def is_lmstudio_available() -> bool:
    """Check if LM Studio server is running."""
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", 1234))
        sock.close()
        return result == 0
    except Exception:
        return False


def is_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available")
def test_claude_code_provider():
    """Test ClaudeCodeProvider with a trivial prompt."""
    from llm_conductor.providers import ClaudeCodeProvider

    provider = ClaudeCodeProvider(model_name="claude-sonnet-4")

    # This test might fail with authentication or configuration issues
    try:
        response = provider.run(TEST_PROMPT)

        # Basic validation - response should be non-empty
        assert response, "Provider returned empty response"
        assert len(response) > 0, "Provider response is empty"
        print(f"\nClaudeCodeProvider response: {response[:100]}...")
    except RuntimeError as e:
        if "exit code" in str(e).lower() or "failed" in str(e).lower():
            pytest.skip(f"Claude CLI authentication or configuration issue: {e}")
        raise


@pytest.mark.integration
@pytest.mark.skipif(not is_openai_codex_available(), reason="codex CLI not available")
def test_openai_codex_provider():
    """Test OpenAICodexProvider with a trivial prompt."""
    from llm_conductor.providers import OpenAICodexProvider

    provider = OpenAICodexProvider(model_name="gpt-5", reasoning_effort="low")

    # This test might fail with usage limit - that's expected
    try:
        response = provider.run(TEST_PROMPT)

        # Basic validation - response should be non-empty
        assert response, "Provider returned empty response"
        assert len(response) > 0, "Provider response is empty"
        print(f"\nOpenAICodexProvider response: {response[:100]}...")
    except RuntimeError as e:
        if "usage limit" in str(e).lower():
            pytest.skip(f"OpenAI Codex usage limit hit: {e}")
        raise


@pytest.mark.integration
@pytest.mark.skipif(not is_openai_configured(), reason="OPENAI_API_KEY not set")
def test_openai_provider():
    """Test OpenAIProvider with a trivial prompt."""
    from llm_conductor.providers import OpenAIProvider

    provider = OpenAIProvider(model_name="gpt-4o-mini")  # Use cheap model for tests

    # OpenAI provider is async-only (LiteLLM-based)
    async def run_async():
        result = await provider.async_run(TEST_PROMPT)
        # async_run returns tuple (content, usage, generation_id)
        return result[0] if isinstance(result, tuple) else result

    response = asyncio.run(run_async())

    # Basic validation - response should be non-empty
    assert response, "Provider returned empty response"
    assert len(response) > 0, "Provider response is empty"
    print(f"\nOpenAIProvider response: {response[:100]}...")


@pytest.mark.integration
@pytest.mark.skipif(not is_openrouter_configured(), reason="OPENROUTER_API_KEY not set")
def test_openrouter_provider():
    """Test OpenRouterProvider with a trivial prompt."""
    from llm_conductor.providers import OpenRouterProvider

    provider = OpenRouterProvider(model_name="openai/gpt-4o-mini")  # Use cheap model

    # OpenRouter provider is async-only (LiteLLM-based)
    async def run_async():
        result = await provider.async_run(TEST_PROMPT)
        return result[0] if isinstance(result, tuple) else result

    response = asyncio.run(run_async())

    # Basic validation - response should be non-empty
    assert response, "Provider returned empty response"
    assert len(response) > 0, "Provider response is empty"
    print(f"\nOpenRouterProvider response: {response[:100]}...")


@pytest.mark.integration
@pytest.mark.skipif(not is_lmstudio_available(), reason="LM Studio server not running")
def test_lmstudio_provider():
    """Test LMStudioProvider with a trivial prompt."""
    from llm_conductor.providers import LMStudioProvider

    provider = LMStudioProvider()

    # LMStudio provider is async-only (LiteLLM-based)
    async def run_async():
        result = await provider.async_run(TEST_PROMPT)
        return result[0] if isinstance(result, tuple) else result

    response = asyncio.run(run_async())

    # Basic validation - response should be non-empty
    assert response, "Provider returned empty response"
    assert len(response) > 0, "Provider response is empty"
    print(f"\nLMStudioProvider response: {response[:100]}...")


@pytest.mark.integration
@pytest.mark.skipif(not is_ollama_available(), reason="Ollama server not running")
def test_ollama_provider():
    """Test OllamaProvider with a trivial prompt."""
    from llm_conductor.providers import OllamaProvider

    provider = OllamaProvider()

    # Ollama provider is async-only (LiteLLM-based)
    async def run_async():
        result = await provider.async_run(TEST_PROMPT)
        return result[0] if isinstance(result, tuple) else result

    response = asyncio.run(run_async())

    # Basic validation - response should be non-empty
    assert response, "Provider returned empty response"
    assert len(response) > 0, "Provider response is empty"
    print(f"\nOllamaProvider response: {response[:100]}...")


@pytest.mark.integration
def test_provider_factory():
    """Test that get_provider factory works for all providers."""
    from llm_conductor import get_provider

    # Test all provider names
    provider_configs = [
        ("claude-code", "claude-sonnet-4", is_claude_code_available()),
        ("openai-codex", "gpt-5", is_openai_codex_available()),
        ("openai", "gpt-4o-mini", is_openai_configured()),
        ("openrouter", "openai/gpt-4o-mini", is_openrouter_configured()),
        ("lmstudio", None, is_lmstudio_available()),
        ("ollama", None, is_ollama_available()),
    ]

    for provider_name, model_name, is_available in provider_configs:
        if not is_available:
            pytest.skip(f"{provider_name} not available")
            continue

        provider = get_provider(provider_name, model_name, reasoning="low")
        assert provider is not None, f"Failed to create {provider_name} provider"
        assert hasattr(provider, "run"), f"{provider_name} provider missing run method"


@pytest.mark.integration
def test_invalid_provider():
    """Test that get_provider raises error for invalid provider name."""
    from llm_conductor import get_provider

    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("invalid-provider")
