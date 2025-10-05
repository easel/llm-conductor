# LLM Conductor

Unified LLM provider orchestration with batch processing and streaming support.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Providers](#supported-providers)
- [Configuration](#configuration)
- [Batch Processing](#batch-processing)
- [Advanced Features](#advanced-features)
- [Logging](#logging)
- [Observability (Optional)](#observability-optional)
  - [MLflow Tracing](#mlflow-tracing)
  - [OpenTelemetry Tracing](#opentelemetry-tracing)
  - [Using Both Simultaneously](#using-both-simultaneously)
  - [Choosing a Backend](#choosing-a-backend)

## Features

- **Multiple Providers**: Support for Claude CLI, OpenAI Codex, OpenAI API, LM Studio, and OpenRouter
- **Batch Processing**: Process multiple prompts in parallel with configurable concurrency
- **Streaming**: Real-time streaming for LiteLLM-based providers
- **Change Detection**: Skip unchanged prompts based on content hashing
- **Auto-continuation**: Automatic handling of truncated responses
- **Retry Logic**: Built-in retry mechanisms for reliability
- **Observability** (Optional): MLflow experiment tracking and OpenTelemetry distributed tracing

## Installation

```bash
# Via UV (recommended)
uv add llm-conductor

# Via pip
pip install llm-conductor

# With observability support (MLflow + OpenTelemetry)
pip install llm-conductor[observability]
```

## Quick Start

### CLI Usage

```bash
# Single prompt from stdin
echo "Explain Python decorators" | llm-conductor run --provider openai

# Batch processing with parallelism
llm-conductor batch \
  --provider openai \
  --model gpt-4 \
  --parallelism 8 \
  --input-dir prompts/ \
  --output-dir outputs/

# List available providers
llm-conductor list
```

### Python API

```python
from llm_conductor import get_provider, run_single_mode, run_batch_mode

# Get a provider instance
provider = get_provider("openai", model_name="gpt-4")

# Run single prompt
response = provider.run("Explain decorators")

# Async execution (LiteLLM providers)
content, usage, gen_id = await provider.async_run("Explain decorators")

# Batch processing
from pathlib import Path
exit_code = run_batch_mode(
    provider=provider,
    parallelism=8,
    verbose=True,
    force=False,
    input_dir=Path("prompts"),
    output_dir=Path("outputs"),
    recursive=False  # Set to True to scan subdirectories
)
```

## Supported Providers

| Provider | Type | Requirements | Default Parallelism |
|----------|------|--------------|---------------------|
| `openai` | API | `OPENAI_API_KEY` | 32 |
| `openai-codex` | CLI | `codex` command | 4 |
| `claude-code` | CLI | `claude` command | 4 |
| `lmstudio` | API | Local LM Studio server | 1 |
| `openrouter` | API | `OPENROUTER_API_KEY` | 32 |

## Configuration

### Environment Variables

```bash
# Provider selection
export MODEL_PROVIDER=openai

# Model configuration
export MODEL_NAME=gpt-4
export MODEL_REASONING=high  # For openai-codex: low, medium, high

# OpenAI configuration
export OPENAI_API_KEY=your-key
export OPENAI_BASE_URL=http://localhost:1234/v1  # For local models
export OPENAI_TEMPERATURE=0.1
export OPENAI_MAX_TOKENS=8192

# LiteLLM configuration
export LITELLM_NUM_RETRIES=3
export LITELLM_TIMEOUT=120
export LITELLM_APP_RETRIES=3
export LITELLM_DEBUG=1  # Enable debug logging

# OpenRouter configuration
export OPENROUTER_API_KEY=your-key

# LM Studio configuration (pre-configured)
# Default: http://localhost:1234/v1

# Observability (optional - requires pip install llm-conductor[observability])
export LITELLM_MLFLOW_ENABLED=true      # Enable MLflow tracing
export MLFLOW_EXPERIMENT_NAME="my-exp"  # MLflow experiment name
export LITELLM_OTEL_ENABLED=true        # Enable OpenTelemetry tracing
export TRACELOOP_BASE_URL="http://localhost:4317"  # OTLP endpoint
```

### Provider-Specific Configuration

#### OpenAI
Supports both cloud and local models via `OPENAI_BASE_URL`.

```bash
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4
```

#### OpenAI Codex
CLI-based provider with reasoning modes.

```bash
# Reasoning levels: low, medium, high
llm-conductor run --provider openai-codex --reasoning high
```

#### Claude Code
CLI-based provider using the `claude` command.

```bash
llm-conductor run --provider claude-code
```

#### LM Studio
Local model server with pre-configured endpoint.

```bash
# Start LM Studio server on default port
llm-conductor batch --provider lmstudio --parallelism 1
```

#### OpenRouter
Unified routing across multiple LLM providers.

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
llm-conductor run --provider openrouter --model anthropic/claude-3.5-sonnet
```

## Batch Processing

### Input Format

Prompts use a naming convention that encodes the expected output format: `{name}.{ext}.prompt.md`

**Naming Convention:**
- `{name}.json.prompt.md` → Outputs `{name}.json` (JSON expected)
- `{name}.md.prompt.md` → Outputs `{name}.md` (Markdown expected)
- `{name}.prompt.md` → Auto-detects format (backward compatibility)

**Example structure:**
```
prompts/
├── summary/
│   └── Table_Summary.md.prompt.md       → Table_Summary.md
├── validation/
│   └── Table_Rules.json.prompt.md       → Table_Rules.json
└── relationship/
    └── relationships.json.prompt.md     → relationships.json
```

### Output Format

**Explicit format hints (recommended):**
- `.json.prompt.md` → Outputs strict JSON (fails if LLM output is invalid)
- `.md.prompt.md` → Outputs markdown with hash comment

**Auto-detection (legacy):**
- Old format `.prompt.md` → Attempts to detect JSON vs markdown
- Less reliable with LLM preamble/postamble

**Hash tracking:**
- JSON dict outputs: `_prompt_hash` embedded in JSON
- Markdown outputs: `<!-- Prompt Hash: {hash} -->` comment
- Used for change detection

### Change Detection

```bash
# Skip unchanged prompts (default)
llm-conductor batch --input-dir prompts/ --output-dir outputs/

# Force regeneration
llm-conductor batch --input-dir prompts/ --output-dir outputs/ --force
```

### Directory Scanning

By default, batch mode scans only the top level of `--input-dir`. Use `--recursive` to scan subdirectories:

```bash
# Non-recursive (default) - only scans prompts/ directory
llm-conductor batch --input-dir prompts/ --output-dir outputs/

# Recursive - scans prompts/ and all subdirectories
llm-conductor batch --input-dir prompts/ --output-dir outputs/ --recursive
# or
llm-conductor batch -r --input-dir prompts/ --output-dir outputs/
```

**Example directory structure:**
```
prompts/
├── task1.prompt.md       ← Processed (both modes)
├── task2.prompt.md       ← Processed (both modes)
└── subfolder/
    └── task3.prompt.md   ← Only processed with --recursive
```

## Advanced Features

### Streaming

LiteLLM providers support streaming in single-prompt mode:

```python
# Stream to stdout
content, usage, gen_id = await provider.async_run(prompt, stream_to_stdout=True)
```

### Auto-continuation

Handles `finish_reason="length"` automatically:

```python
# Automatically continues truncated responses
provider = get_provider("openai", model_name="gpt-4")
content, usage, gen_id = await provider.async_run(long_prompt)
```

### Custom Parallelism

```bash
# Default parallelism based on provider
llm-conductor batch --provider openai --input-dir prompts/ --output-dir outputs/

# Custom parallelism
llm-conductor batch --provider openai -j 16 --input-dir prompts/ --output-dir outputs/
```

## Logging

Logs are written to `.local/logs/llm-conductor.log`:

```bash
# View recent errors
tail -f .local/logs/llm-conductor.log | grep ERROR

# Debug mode
export LITELLM_DEBUG=1
llm-conductor batch --verbose --provider openai
```

## Observability (Optional)

llm-conductor supports two observability backends for tracing LLM calls:

1. **MLflow** - LLM-optimized experiment tracking with UI, cost analysis, and model comparison
2. **OpenTelemetry** - Distributed tracing compatible with Jaeger, Grafana, Datadog, etc.

Both can be enabled simultaneously.

### Quick Start

```bash
# Install with observability support
pip install llm-conductor[observability]

# Option 1: MLflow (local experiment tracking)
export LITELLM_MLFLOW_ENABLED=true
llm-conductor batch --provider openai --input-dir prompts/ --output-dir outputs/
mlflow ui  # View at http://localhost:5000

# Option 2: OpenTelemetry (production monitoring)
export LITELLM_OTEL_ENABLED=true
export TRACELOOP_BASE_URL="http://jaeger:4317"
llm-conductor batch --provider openai --input-dir prompts/ --output-dir outputs/

# Option 3: Both simultaneously
export LITELLM_MLFLOW_ENABLED=true
export LITELLM_OTEL_ENABLED=true
export TRACELOOP_BASE_URL="http://jaeger:4317"
llm-conductor batch --provider openai --input-dir prompts/ --output-dir outputs/
```

### Installation

```bash
# With pip
pip install llm-conductor[observability]

# With uv
uv add llm-conductor --optional observability
```

This installs both MLflow and OpenTelemetry dependencies.

---

## MLflow Tracing

**Best for:** LLM experiment tracking, cost analysis, prompt comparison, local development

MLflow provides native liteLLM integration with automatic tracing of all LLM calls.

### Setup

```bash
# Enable MLflow tracing (local file storage)
export LITELLM_MLFLOW_ENABLED=true
export MLFLOW_EXPERIMENT_NAME="my-llm-experiment"

# Run batch processing - traces stored in ./mlruns
llm-conductor batch --provider openai --input-dir prompts/ --output-dir outputs/

# View traces in MLflow UI
mlflow ui
# Open http://localhost:5000
```

### Configuration

```bash
# Local file storage (default)
export LITELLM_MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI="./mlruns"
export MLFLOW_EXPERIMENT_NAME="llm-conductor"

# Remote tracking server
export LITELLM_MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
export MLFLOW_EXPERIMENT_NAME="production-llm"

# Databricks workspace
export LITELLM_MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI="databricks"
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"
```

### What Gets Tracked

MLflow automatically logs:
- **Prompts and responses** - Full text of inputs and outputs
- **Token usage** - Prompt, completion, and total tokens
- **Cost tracking** - Per-request cost estimates
- **Model parameters** - Temperature, max_tokens, model name
- **Timing** - Request duration and timestamps
- **Errors** - Failed requests with error messages

### MLflow UI Features

- **Experiment comparison** - Compare different prompts, models, or parameters side-by-side
- **Cost analysis** - Aggregate costs across runs
- **Search and filter** - Find specific runs by model, cost, duration, etc.
- **Export** - Download traces as CSV, JSON, or parquet

### Example: Running Multiple Experiments

```bash
# Experiment 1: GPT-4
export LITELLM_MLFLOW_ENABLED=true
export MLFLOW_EXPERIMENT_NAME="gpt4-baseline"
llm-conductor batch --provider openai --model gpt-4 \
  --input-dir prompts/ --output-dir outputs/gpt4/

# Experiment 2: GPT-4o-mini (cost comparison)
export MLFLOW_EXPERIMENT_NAME="gpt4o-mini-test"
llm-conductor batch --provider openai --model gpt-4o-mini \
  --input-dir prompts/ --output-dir outputs/gpt4o-mini/

# Compare results in UI
mlflow ui
```

---

## OpenTelemetry Tracing

**Best for:** Production observability, multi-service tracing, existing OTLP infrastructure

### Configuration

```bash
# Enable tracing
export LITELLM_OTEL_ENABLED=true

# OTLP endpoint (required if enabled)
export TRACELOOP_BASE_URL="http://localhost:4317"

# Optional: Sampling rate (0.0 to 1.0, default: 1.0)
export TRACELOOP_TRACE_RATE=1.0

# Optional: Service name (default: llm-conductor)
export OTEL_SERVICE_NAME="llm-conductor"

# Optional: Disable anonymous telemetry
export TRACELOOP_TELEMETRY_ENABLED=false
```

### What Gets Traced

OpenLLMetry automatically captures:
- **LLM API calls** - All liteLLM completions (OpenAI, Anthropic, etc.)
- **Token usage** - Prompt tokens, completion tokens, total tokens
- **Cost tracking** - Per-request cost estimates
- **Streaming progress** - Real-time streaming and continuations
- **Retries & errors** - Automatic retry attempts and failure reasons
- **Model parameters** - Temperature, max_tokens, top_p, etc.

### Supported Backends

**Open Source:**
- Jaeger (distributed tracing)
- Zipkin
- Grafana Tempo

**Commercial:**
- Datadog APM
- New Relic
- Honeycomb
- Dynatrace
- Splunk

**LLM-Specific:**
- Traceloop (hosted platform)
- Langfuse
- Phoenix (Arize AI)

### Example: Jaeger Setup

```bash
# Start Jaeger locally
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# Configure llm-conductor
export LITELLM_OTEL_ENABLED=true
export TRACELOOP_BASE_URL="http://localhost:4317"

# Run batch processing (traces sent to Jaeger)
llm-conductor batch --provider openai --input-dir prompts/ --output-dir outputs/

# View traces at http://localhost:16686
```

### Trace Example

Each LLM call generates a span with:
- `gen_ai.system` = "openai"
- `gen_ai.request.model` = "gpt-4"
- `gen_ai.usage.input_tokens` = 145
- `gen_ai.usage.output_tokens` = 892
- `gen_ai.response.finish_reasons` = ["stop"]
- Cost, latency, and metadata

---

## Using Both Simultaneously

MLflow and OpenTelemetry can run together:

```bash
# Enable both backends
export LITELLM_MLFLOW_ENABLED=true
export MLFLOW_EXPERIMENT_NAME="experiment-1"

export LITELLM_OTEL_ENABLED=true
export TRACELOOP_BASE_URL="http://localhost:4317"

# Traces sent to both MLflow and OTLP collector
llm-conductor batch --provider openai --input-dir prompts/ --output-dir outputs/
```

This is useful for:
- Local development with MLflow UI + production monitoring with Datadog/Grafana
- Short-term experiment tracking (MLflow) + long-term archival (OTLP → S3)
- Different teams using different observability platforms

---

## Choosing a Backend

| Feature | MLflow | OpenTelemetry |
|---------|--------|---------------|
| **Setup** | Very simple | Requires collector |
| **UI** | Built-in | Separate (Jaeger/Grafana) |
| **LLM-specific** | ✅ Optimized | Generic spans |
| **Cost tracking** | ✅ Aggregated | Per-span only |
| **Experiment comparison** | ✅ Native | Manual |
| **Prompt versioning** | ✅ Built-in | ❌ |
| **Multi-service tracing** | Limited | ✅ Full support |
| **Production monitoring** | Good | ✅ Excellent |
| **Data retention** | File/DB | Configurable backend |

**Use MLflow if:**
- You're experimenting with prompts or models
- You want cost analysis and comparison
- You need a UI without extra setup
- You're doing local development

**Use OpenTelemetry if:**
- You have existing OTLP infrastructure
- You need multi-service distributed tracing
- You're monitoring production systems
- You want flexibility in storage backends

## License

Apache 2.0 License - see LICENSE file for details.
