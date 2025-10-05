# LLM Conductor - Claude Code Guide

Unified LLM provider orchestration with batch processing and streaming support.

## Quick Start

```bash
# View all available commands
make help

# Setup development environment
make install-dev

# Run the CLI
make run ARGS="list"
make run ARGS="batch --provider openai --input-dir prompts/ --output-dir outputs/"

# Run tests
make test
```

## Project Structure

**Core Implementation:**
- `src/llm_conductor/base.py` - Base provider interface
- `src/llm_conductor/litellm_base.py` - LiteLLM provider base class
- `src/llm_conductor/cli_base.py` - CLI provider base class
- `src/llm_conductor/runner.py` - Batch processing engine
- `src/llm_conductor/cli.py` - CLI entry point

**Providers:**
- `src/llm_conductor/providers/openai.py` - OpenAI API (cloud + local)
- `src/llm_conductor/providers/claude_code.py` - Claude CLI wrapper
- `src/llm_conductor/providers/openai_codex.py` - OpenAI Codex CLI
- `src/llm_conductor/providers/lmstudio.py` - LM Studio local server
- `src/llm_conductor/providers/ollama.py` - Ollama local server
- `src/llm_conductor/providers/openrouter.py` - OpenRouter API

**Observability:**
- `src/llm_conductor/observability.py` - OpenTelemetry integration
- `src/llm_conductor/mlflow_integration.py` - MLflow experiment tracking

## Key Technologies

- **Python 3.12** (strict requirement)
- **uv** - Fast Python package manager
- **LiteLLM** - Unified LLM API interface
- **Click** - CLI framework
- **pytest** - Testing framework

## Development Notes

- Uses `uv` for dependency management (faster than pip)
- Optional observability extras: `make install-observability`
- Logs written to `.local/logs/llm-conductor.log`
- See README.md for detailed configuration and usage
