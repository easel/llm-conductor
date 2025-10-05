# Contributing to LLM Conductor

Thank you for your interest in contributing to **LLM Conductor**! This guide is designed for **human contributors** who want to improve the project through code, documentation, or feedback.

We welcome contributions of all kinds:
- 🐛 Bug fixes
- ✨ New features (especially new provider implementations)
- 📚 Documentation improvements
- 🧪 Test coverage enhancements
- 💡 Feature requests and design discussions

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive in all interactions
- Assume good faith and intent from other contributors
- Focus on what is best for the project and community
- Accept constructive criticism gracefully

## Getting Started

### Prerequisites

- **Python 3.12** (strictly required - see `pyproject.toml`)
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))
- **git** for version control
- Familiarity with async Python and CLI tools is helpful

### Initial Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/llm-conductor.git
   cd llm-conductor
   ```

2. **Install development dependencies:**
   ```bash
   make install-dev
   ```

3. **Verify your setup:**
   ```bash
   make test
   make list-providers
   ```

4. **Optional: Install observability extras for tracing features:**
   ```bash
   make install-observability
   ```

### Project Structure

```
llm-conductor/
├── src/llm_conductor/
│   ├── base.py              # Base provider interface
│   ├── litellm_base.py      # LiteLLM provider base class
│   ├── cli_base.py          # CLI provider base class
│   ├── runner.py            # Batch processing engine
│   ├── cli.py               # CLI entry point
│   ├── observability.py     # OpenTelemetry integration
│   ├── mlflow_integration.py # MLflow tracing
│   └── providers/           # Provider implementations
│       ├── openai.py
│       ├── claude_code.py
│       ├── openai_codex.py
│       ├── lmstudio.py
│       ├── ollama.py
│       └── openrouter.py
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── Makefile                 # Development commands
├── pyproject.toml           # Package configuration
└── README.md                # User documentation
```

## Development Workflow

### Using the Makefile

The project includes a comprehensive Makefile with common development tasks:

```bash
make help               # Display all available commands
make install-dev        # Install with dev dependencies
make test               # Run all tests
make test-unit          # Run only unit tests
make test-integration   # Run integration tests
make test-verbose       # Run tests with verbose output
make run ARGS="list"    # Run the CLI with arguments
make build              # Build the package
make clean              # Remove build artifacts
```

### Running the CLI During Development

```bash
# List providers
make run ARGS="list"

# Test single prompt
echo "Explain Python decorators" | make run ARGS="run --provider openai"

# Test batch processing
make run ARGS="batch --provider openai --input-dir prompts/ --output-dir outputs/ --verbose"
```

### Logs and Debugging

Logs are written to `.local/logs/llm-conductor.log`:

```bash
# Watch logs in real-time
tail -f .local/logs/llm-conductor.log

# Filter for errors
tail -f .local/logs/llm-conductor.log | grep ERROR

# Enable debug mode
export LITELLM_DEBUG=1
make run ARGS="batch --verbose --provider openai"
```

## Making Contributions

### 1. Create a Branch

Use descriptive branch names:

```bash
git checkout -b feature/add-gemini-provider
git checkout -b fix/batch-processing-race-condition
git checkout -b docs/improve-provider-guide
```

### 2. Make Your Changes

- Write clear, concise code following existing patterns
- Add tests for new functionality
- Update documentation if needed
- Follow Python conventions (PEP 8)

### 3. Test Your Changes

Run the full test suite before submitting:

```bash
make test
```

For specific test types:

```bash
make test-unit          # Fast unit tests only
make test-integration   # Slower integration tests
make test-verbose       # See detailed output
```

### 4. Commit Your Changes

Write clear commit messages:

```bash
git add .
git commit -m "feat: add Google Gemini provider

- Implement GeminiProvider class extending LiteLLMProvider
- Add Gemini-specific configuration options
- Include unit and integration tests
- Update README with Gemini setup instructions"
```

**Commit message guidelines:**
- Use conventional commit format: `type: description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
- Keep first line under 72 characters
- Add detailed explanation in body if needed

### 5. Push and Create a Pull Request

```bash
git push origin feature/add-gemini-provider
```

Then open a pull request on GitHub with:
- **Clear title** describing the change
- **Description** explaining what and why (not just how)
- **Testing notes** - how you verified the changes
- **Related issues** - reference any GitHub issues (e.g., "Fixes #123")

**Pull request template:**

```markdown
## Summary
Brief description of changes

## Changes Made
- Bullet point list of specific changes
- Include any breaking changes

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] Documentation updated

## Related Issues
Fixes #<issue-number>
```

## Testing Guidelines

### Writing Tests

Tests use **pytest** and are organized by type:

**Unit Tests** (`tests/unit/`):
- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution (< 1 second per test)

**Integration Tests** (`tests/integration/`):
- Test provider integrations end-to-end
- May require API keys or running services
- Slower execution

### Test Structure Example

```python
import pytest
from llm_conductor.providers.openai import OpenAIProvider

class TestOpenAIProvider:
    """Test OpenAI provider functionality."""

    def test_initialization(self):
        """Test provider initializes with correct defaults."""
        provider = OpenAIProvider(model_name="gpt-4")
        assert provider.model_name == "gpt-4"
        assert provider.default_parallelism == 32

    @pytest.mark.asyncio
    async def test_async_run(self, mock_openai_response):
        """Test async execution returns expected format."""
        provider = OpenAIProvider(model_name="gpt-4")
        content, usage, gen_id = await provider.async_run("Test prompt")
        assert isinstance(content, str)
        assert usage > 0
        assert gen_id is not None
```

### Running Specific Tests

```bash
# Run a specific test file
uv run pytest tests/unit/test_runner.py

# Run a specific test function
uv run pytest tests/unit/test_runner.py::test_batch_processing

# Run with coverage
uv run pytest --cov=llm_conductor --cov-report=html
```

## Code Style and Standards

### Python Conventions

- **Python 3.12** features are allowed
- Use **type hints** for function signatures
- Follow **PEP 8** naming conventions
- Write **docstrings** for public functions and classes

```python
async def async_run(
    self,
    prompt: str,
    stream_to_stdout: bool = False,
    continuation_context: Optional[str] = None
) -> tuple[str, int, str]:
    """Execute prompt asynchronously with streaming support.

    Args:
        prompt: The input prompt text
        stream_to_stdout: Whether to stream output to stdout
        continuation_context: Previous context for auto-continuation

    Returns:
        Tuple of (content, token_usage, generation_id)

    Raises:
        ProviderError: If the API request fails
    """
```

### Project Conventions

- **Error Handling**: Use try/except blocks with specific exception types
- **Logging**: Use the configured logger from `src/llm_conductor/base.py`
- **Configuration**: Support both environment variables and constructor args
- **Async**: Use `async`/`await` for I/O operations in LiteLLM providers

### Code Organization

```python
# Standard library imports first
import os
import asyncio
from typing import Optional

# Third-party imports
import click
from litellm import acompletion

# Local imports
from llm_conductor.base import BaseProvider
from llm_conductor.litellm_base import LiteLLMProvider
```

## Adding a New Provider

Providers are the core extensibility point. Here's how to add one:

### 1. Choose the Right Base Class

- **LiteLLMProvider** (`litellm_base.py`) - For API-based providers supported by LiteLLM
- **CLIProvider** (`cli_base.py`) - For CLI-based tools (like Claude Code)
- **BaseProvider** (`base.py`) - For custom implementations

### 2. Create Provider File

```python
# src/llm_conductor/providers/newprovider.py

from llm_conductor.litellm_base import LiteLLMProvider

class NewProvider(LiteLLMProvider):
    """Provider for New LLM API."""

    DEFAULT_MODEL = "new-model-v1"
    DEFAULT_PARALLELISM = 32

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize New provider.

        Args:
            model_name: Model to use (default: new-model-v1)
            api_key: API key (default: from NEW_API_KEY env var)
        """
        super().__init__(
            provider_name="new",
            model_name=model_name or self.DEFAULT_MODEL,
            api_key=api_key or os.getenv("NEW_API_KEY"),
            **kwargs
        )
```

### 3. Register in `providers/__init__.py`

```python
from llm_conductor.providers.newprovider import NewProvider

PROVIDERS = {
    "openai": OpenAIProvider,
    "new": NewProvider,  # Add your provider
    # ... other providers
}
```

### 4. Add Tests

Create `tests/unit/test_newprovider.py` and `tests/integration/test_newprovider.py`.

### 5. Update Documentation

Add provider details to README.md:
- Supported Providers table
- Configuration section
- Usage examples

## Documentation

### Updating Documentation

- **README.md** - User-facing documentation
- **CLAUDE.md** - Claude Code development guide
- **Docstrings** - Inline code documentation
- **Type hints** - Function signatures

### Documentation Standards

- Use clear, concise language
- Include code examples for complex features
- Keep examples up-to-date with code changes
- Use markdown formatting consistently

### Building Documentation Locally

Currently, the project uses markdown files in the repository. If Sphinx or similar is added:

```bash
# Future: Generate HTML docs
make docs
```

## Reporting Issues

### Bug Reports

When reporting bugs, include:
- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs. actual behavior
- **Environment details**:
  - Python version (`python --version`)
  - Package version (`uv run python -c "from importlib.metadata import version; print(version('llm-conductor'))"`)
  - OS and platform
- **Logs** from `.local/logs/llm-conductor.log`
- **Minimal reproducible example** if possible

### Feature Requests

For new features:
- Describe the problem or use case
- Propose a solution if you have one
- Discuss potential implementation approaches
- Consider backward compatibility

### Security Issues

For security vulnerabilities:
- **Do NOT** open a public GitHub issue
- Email the maintainers directly (check repository for contact)
- Provide detailed information about the vulnerability

## License Agreement

By contributing to LLM Conductor, you agree that your contributions will be licensed under the **Apache License 2.0**.

As stated in Section 5 of the LICENSE file:

> Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions.

This means:
- You retain copyright to your contributions
- You grant the project a perpetual, worldwide license to use your contributions
- Your contributions must not include code with incompatible licenses
- You confirm you have the right to submit the contribution

## Getting Help

### Community Resources

- **GitHub Issues**: https://github.com/easel/llm-conductor/issues
- **Pull Requests**: https://github.com/easel/llm-conductor/pulls
- **Discussions**: Use GitHub Discussions for questions and ideas

### Before Asking

1. Check existing issues and pull requests
2. Review the README.md and code documentation
3. Try debugging with `--verbose` and `LITELLM_DEBUG=1`
4. Search logs in `.local/logs/llm-conductor.log`

### Asking Good Questions

- Provide context and what you've tried
- Include relevant code snippets and error messages
- Mention your environment (OS, Python version, etc.)
- Be patient and respectful

---

## Thank You!

Your contributions make LLM Conductor better for everyone. Whether you're fixing a typo, adding a feature, or reporting a bug, we appreciate your time and effort.

Happy contributing! 🎉
