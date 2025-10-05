"""MLflow integration for LLM tracing (alternative to OpenTelemetry)."""

from __future__ import annotations

import logging
import os

# Gracefully attempt to import mlflow (optional dependency)
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

__all__ = ["MLFLOW_AVAILABLE", "init_mlflow_tracing"]


def init_mlflow_tracing() -> bool:
    """Initialize MLflow auto-tracing for liteLLM.

    MLflow provides native liteLLM integration via mlflow.litellm.autolog(),
    which automatically logs all LLM calls with rich metadata including:
    - Prompts and responses
    - Token usage and costs
    - Model parameters
    - Timing information

    Environment Variables
    ---------------------
    LITELLM_MLFLOW_ENABLED : str
        Set to "true" to enable MLflow tracing (default: disabled)
    MLFLOW_TRACKING_URI : str
        MLflow tracking server URI (default: ./mlruns - local file storage)

    Examples:
        - "./mlruns" (local file storage)
        - "http://mlflow-server:5000" (remote tracking server)
        - "databricks" (Databricks workspace)
    MLFLOW_EXPERIMENT_NAME : str
        Experiment name for grouping runs (default: llm-conductor)
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING : str
        Set to "false" to disable system metrics logging (default: true)

    Returns:
    -------
        True if MLflow tracing was successfully enabled, False otherwise

    Examples:
    --------
        # Local file storage (default)
        export LITELLM_MLFLOW_ENABLED=true
        export MLFLOW_EXPERIMENT_NAME="my-experiment"

        # Remote tracking server
        export LITELLM_MLFLOW_ENABLED=true
        export MLFLOW_TRACKING_URI="http://mlflow:5000"
        export MLFLOW_EXPERIMENT_NAME="production-llm"

        # Databricks workspace
        export LITELLM_MLFLOW_ENABLED=true
        export MLFLOW_TRACKING_URI="databricks"
        export DATABRICKS_HOST="https://your-workspace.databricks.com"
        export DATABRICKS_TOKEN="your-token"

    """
    logger = logging.getLogger("llm_conductor")

    # Check if user wants MLflow tracing enabled
    mlflow_enabled = os.getenv("LITELLM_MLFLOW_ENABLED", "").lower() == "true"

    if not mlflow_enabled:
        logger.debug(
            "MLflow tracing disabled (LITELLM_MLFLOW_ENABLED not set to 'true')"
        )
        return False

    if not MLFLOW_AVAILABLE:
        logger.warning(
            "⚠️  MLflow tracing requested but MLflow not installed. "
            "Install with: pip install llm-conductor[observability]"
        )
        return False

    try:
        # Set tracking URI (default: local file storage)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        # Set experiment (creates if doesn't exist)
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "llm-conductor")
        mlflow.set_experiment(experiment_name)

        # Enable liteLLM auto-logging
        # This instruments all liteLLM calls with tracing
        mlflow.litellm.autolog()

        logger.info("✓ MLflow tracing enabled for liteLLM")
        logger.debug(f"   Tracking URI: {tracking_uri}")
        logger.debug(f"   Experiment: {experiment_name}")

        return True
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize MLflow tracing: {e}")
        logger.debug(f"   Error details: {e}", exc_info=True)
        return False
