# LLM Conductor Integration Tests

Optional integration tests that verify each model provider can process a trivial prompt.

## Running Tests

### Run all available providers:
```bash
pytest packages/llm-conductor/tests/integration/ -v
```

### Run specific provider:
```bash
pytest packages/llm-conductor/tests/integration/test_providers.py::test_claude_code_provider -v
```

### Run with markers:
```bash
# Run only integration tests
pytest -m integration packages/llm-conductor/tests/integration/

# Skip integration tests (for CI)
pytest -m "not integration"
```

## Test Requirements

Tests are automatically skipped if the provider is not available. Configure providers before running:

### claude-code
Requires `claude` CLI installed and authenticated:
```bash
which claude  # Should return path
claude --version  # Should work
```

### openai-codex
Requires `codex` CLI installed and authenticated:
```bash
which codex  # Should return path
codex --help  # Should work
```

**Note**: Tests may skip if usage limit is hit (expected behavior).

### openai
Requires OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

### openrouter
Requires OpenRouter API key:
```bash
export OPENROUTER_API_KEY=sk-or-...
```

### lmstudio
Requires LM Studio server running:
```bash
# Start LM Studio and load a model
# Server should be accessible at http://localhost:1234
```

### ollama
Requires Ollama server running:
```bash
# Start Ollama server
ollama serve
# Server should be accessible at http://localhost:11434

# Load a model (if not already loaded)
ollama pull gpt-oss:120b
```

## What Gets Tested

Each provider test:
1. Creates a provider instance with appropriate configuration
2. Sends a trivial JSON generation prompt
3. Validates the response is non-empty
4. Handles expected failures (usage limits, missing config) gracefully

## CI/CD Integration

Add to your CI config to run tests only for available providers:
```yaml
# GitHub Actions example
- name: Run provider integration tests
  run: pytest -m integration packages/llm-conductor/tests/integration/ -v
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  continue-on-error: true  # Don't fail CI if providers unavailable
```
