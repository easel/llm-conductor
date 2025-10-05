"""CLI for llm-conductor - Unified LLM provider orchestration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from llm_conductor import (
    get_provider,
    init_observability,
    run_batch_mode,
    run_single_mode,
)


@click.group()
@click.version_option()
def main():
    """LLM Conductor - Unified LLM provider orchestration with batch processing."""
    # Initialize OpenTelemetry tracing if enabled
    init_observability()


@main.command(name="run")
@click.option(
    "--provider",
    default=lambda: os.getenv("MODEL_PROVIDER", "openai"),
    help="Model provider to use (default: openai)",
)
@click.option(
    "--model",
    default=lambda: os.getenv("MODEL_NAME"),
    help="Model name (default: provider-specific default)",
)
@click.option(
    "--reasoning",
    type=click.Choice(["low", "medium", "high"]),
    default=lambda: os.getenv("MODEL_REASONING"),
    help="Reasoning effort level (openai-codex only)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def run_command(provider: str, model: str | None, reasoning: str, verbose: bool):
    """Run single prompt from stdin and write to stdout."""
    try:
        provider_instance = get_provider(provider, model, reasoning)
        exit_code = run_single_mode(provider_instance, verbose)
        sys.exit(exit_code)
    except Exception as e:
        click.echo(f"❌ Run failed: {e}", err=True)
        sys.exit(1)


@main.command(name="batch")
@click.option(
    "--provider",
    default=lambda: os.getenv("MODEL_PROVIDER", "openai"),
    help="Model provider to use (default: openai)",
)
@click.option(
    "--model",
    default=lambda: os.getenv("MODEL_NAME"),
    help="Model name (default: provider-specific default)",
)
@click.option(
    "--reasoning",
    type=click.Choice(["low", "medium", "high"]),
    default=lambda: os.getenv("MODEL_REASONING", "high"),
    help="Reasoning effort level (default: high, openai-codex only)",
)
@click.option(
    "-j",
    "--parallelism",
    type=int,
    default=None,
    help="Number of parallel workers (default: 1 for lmstudio, 4 for openai/claude-code, 32 for others)",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Base directory for input prompts",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Base directory for output files",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--force",
    is_flag=True,
    help="Force regeneration of all outputs (skip change detection)",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Recursively scan subdirectories for .prompt.md files",
)
def batch_command(
    provider: str,
    model: str | None,
    reasoning: str,
    parallelism: int | None,
    input_dir: Path,
    output_dir: Path,
    verbose: bool,
    force: bool,
    recursive: bool,
):
    """Run batch processing of prompts with parallelism."""
    try:
        provider_instance = get_provider(provider, model, reasoning)

        # Determine parallelism if not explicitly set
        if parallelism is None:
            if provider == "lmstudio":
                parallelism = 1
            elif provider in ("claude-code", "openai-codex"):
                parallelism = 4  # Lower parallelism for CLI providers
            else:
                parallelism = 32

        # Build prompt files list: stdin or directory scan
        prompt_files = []
        if not sys.stdin.isatty():
            # Parse stdin for relative paths
            stdin_lines = [line.strip() for line in sys.stdin if line.strip()]
            for line in stdin_lines:
                prompt_file = input_dir / line
                prompt_files.append(prompt_file)
        # Directory scan (recursive or non-recursive)
        elif recursive:
            prompt_files = sorted(input_dir.rglob("*.prompt.md"))
        else:
            prompt_files = sorted(input_dir.glob("*.prompt.md"))

        exit_code = run_batch_mode(
            provider_instance,
            prompt_files,
            parallelism,
            verbose,
            force,
            input_dir,
            output_dir,
        )
        sys.exit(exit_code)
    except Exception as e:
        click.echo(f"❌ Batch run failed: {e}", err=True)
        sys.exit(1)


@main.command(name="list")
def list_command():
    """List available LLM providers."""
    providers = {
        "claude-code": "Claude CLI (requires claude command)",
        "openai-codex": "OpenAI Codex CLI (requires codex command)",
        "openai": "OpenAI API (supports local models via OPENAI_BASE_URL)",
        "lmstudio": "LM Studio local server",
        "openrouter": "OpenRouter API (unified LLM routing)",
    }

    click.echo("Available LLM Providers:")
    click.echo()
    for name, description in providers.items():
        click.echo(f"  {name:15s} - {description}")


if __name__ == "__main__":
    main()
