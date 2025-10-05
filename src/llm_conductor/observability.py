"""Observability integration for LLM tracing (OpenTelemetry and MLflow support)."""

from __future__ import annotations

import logging
import os

# Gracefully attempt to import traceloop-sdk (optional dependency)
try:
    from traceloop.sdk import Traceloop

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Import MLflow integration
from llm_conductor.mlflow_integration import MLFLOW_AVAILABLE, init_mlflow_tracing

__all__ = ["MLFLOW_AVAILABLE", "OTEL_AVAILABLE", "init_observability"]


def init_opentelemetry_tracing() -> bool:
    """Initialize OpenTelemetry tracing if enabled and available.

    Checks the LITELLM_OTEL_ENABLED environment variable and initializes
    OpenLLMetry (Traceloop SDK) if:
    1. The observability dependencies are installed
    2. The LITELLM_OTEL_ENABLED env var is set to "true" (case-insensitive)

    Environment Variables
    ---------------------
    LITELLM_OTEL_ENABLED : str
        Set to "true" to enable OpenTelemetry tracing (default: disabled)
    TRACELOOP_BASE_URL : str
        OTLP endpoint URL (e.g., "http://localhost:4317")
    TRACELOOP_HEADERS : str
        JSON-encoded headers for authentication
    TRACELOOP_TRACE_RATE : float
        Sampling rate (0.0 to 1.0, default: 1.0 = trace 100%)
    TRACELOOP_TELEMETRY_ENABLED : str
        Set to "false" to disable anonymous telemetry

    Returns
    -------
        True if tracing was successfully enabled, False otherwise

    Examples
    --------
        # Enable tracing with Jaeger
        export LITELLM_OTEL_ENABLED=true
        export TRACELOOP_BASE_URL="http://localhost:4317"

        # Enable with sampling (trace 10% of requests)
        export LITELLM_OTEL_ENABLED=true
        export TRACELOOP_BASE_URL="http://localhost:4317"
        export TRACELOOP_TRACE_RATE=0.1

    """
    logger = logging.getLogger("llm_conductor")

    # Check if user wants tracing enabled
    otel_enabled = os.getenv("LITELLM_OTEL_ENABLED", "").lower() == "true"

    if not otel_enabled:
        logger.debug(
            "OpenTelemetry tracing disabled (LITELLM_OTEL_ENABLED not set to 'true')"
        )
        return False

    if not OTEL_AVAILABLE:
        logger.warning(
            "⚠️  OpenTelemetry tracing requested but dependencies not installed. "
            "Install with: pip install llm-conductor[observability]"
        )
        return False

    # Initialize Traceloop SDK
    try:
        Traceloop.init(
            app_name=os.getenv("OTEL_SERVICE_NAME", "llm-conductor"),
            disable_batch=True,  # Send traces immediately for real-time observability
            exporter_type="otlp",  # Use OpenTelemetry Protocol
        )
        logger.info("✓ OpenTelemetry tracing enabled via OpenLLMetry")
        logger.debug(f"   OTLP endpoint: {os.getenv('TRACELOOP_BASE_URL', 'default')}")
        logger.debug(
            f"   Service name: {os.getenv('OTEL_SERVICE_NAME', 'llm-conductor')}"
        )
        return True
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize OpenTelemetry tracing: {e}")
        return False


def init_observability() -> bool:
    """Initialize observability tracing (MLflow or OpenTelemetry).

    Supports both MLflow and OpenTelemetry tracing backends. Priority:
    1. MLflow (if LITELLM_MLFLOW_ENABLED=true)
    2. OpenTelemetry (if LITELLM_OTEL_ENABLED=true)
    3. None (if neither enabled)

    Both can be enabled simultaneously - traces will be sent to both backends.

    Returns
    -------
        True if any tracing backend was enabled, False otherwise

    """
    mlflow_enabled = init_mlflow_tracing()
    otel_enabled = init_opentelemetry_tracing()

    return mlflow_enabled or otel_enabled
