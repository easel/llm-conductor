"""LLM Conductor - Unified LLM provider orchestration with batch processing.

Supports multiple LLM providers through a unified interface:
- claude-code: Claude CLI (claude -p)
- openai-codex: OpenAI Codex CLI (codex exec)
- openai: OpenAI SDK (supports local models via OPENAI_BASE_URL)
- lmstudio: Pre-configured LM Studio server
- ollama: Ollama local server (localhost:11434)
- openrouter: OpenRouter API (unified LLM routing)
"""

from __future__ import annotations

from .base import ModelProvider
from .cli_base import CLIProvider
from .mlflow_integration import MLFLOW_AVAILABLE, init_mlflow_tracing
from .observability import OTEL_AVAILABLE, init_observability
from .providers import (
    ClaudeCodeProvider,
    LMStudioProvider,
    OllamaProvider,
    OpenAICodexProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from .runner import (
    async_process_single_task,
    extract_existing_hash,
    process_single_task,
    run_batch_mode,
    run_single_mode,
)

__all__ = [
    "MLFLOW_AVAILABLE",
    "OTEL_AVAILABLE",
    "CLIProvider",
    "ClaudeCodeProvider",
    "LMStudioProvider",
    "ModelProvider",
    "OllamaProvider",
    "OpenAICodexProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "async_process_single_task",
    "extract_existing_hash",
    "get_provider",
    "init_mlflow_tracing",
    "init_observability",
    "process_single_task",
    "run_batch_mode",
    "run_single_mode",
]


def get_provider(
    provider_name: str, model_name: str | None = None, reasoning: str | None = None
) -> ModelProvider:
    """Get the appropriate provider instance.

    Args:
    ----
        provider_name: Name of the provider (claude-code, openai-codex, openai, lmstudio, ollama, openrouter)
        model_name: Optional model name override
        reasoning: Reasoning effort level (low, medium, high) - only used by some providers

    Returns:
    -------
        ModelProvider instance

    Raises:
    ------
        ValueError: If provider_name is not recognized

    """
    providers = {
        "claude-code": ClaudeCodeProvider,
        "openai-codex": OpenAICodexProvider,
        "openai": OpenAIProvider,
        "lmstudio": LMStudioProvider,
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider,
    }

    provider_class = providers.get(provider_name)
    if not provider_class:
        msg = f"Unknown provider: {provider_name}. Valid providers: {', '.join(providers.keys())}"
        raise ValueError(msg)

    # Pass reasoning to providers that support it
    if provider_name == "openai-codex" and reasoning:
        return provider_class(model_name, reasoning_effort=reasoning)
    return provider_class(model_name)
