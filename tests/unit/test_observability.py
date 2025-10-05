"""Unit tests for observability integrations."""

from __future__ import annotations

from unittest.mock import patch


class TestMLflowIntegration:
    """Test MLflow integration functions."""

    def test_init_mlflow_tracing_disabled_by_default(self, monkeypatch):
        """Test that MLflow tracing is disabled by default."""
        from llm_conductor.mlflow_integration import init_mlflow_tracing

        # Don't set LITELLM_MLFLOW_ENABLED
        result = init_mlflow_tracing()

        assert result is False

    @patch("llm_conductor.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_init_mlflow_tracing_not_installed(self, monkeypatch):
        """Test MLflow tracing when MLflow not installed."""
        from llm_conductor.mlflow_integration import init_mlflow_tracing

        monkeypatch.setenv("LITELLM_MLFLOW_ENABLED", "true")

        result = init_mlflow_tracing()

        assert result is False

    def test_init_mlflow_tracing_false_value(self, monkeypatch):
        """Test that LITELLM_MLFLOW_ENABLED=false disables tracing."""
        from llm_conductor.mlflow_integration import init_mlflow_tracing

        monkeypatch.setenv("LITELLM_MLFLOW_ENABLED", "false")

        result = init_mlflow_tracing()

        assert result is False

    def test_mlflow_available_constant(self):
        """Test that MLFLOW_AVAILABLE constant is available."""
        from llm_conductor.mlflow_integration import MLFLOW_AVAILABLE

        assert isinstance(MLFLOW_AVAILABLE, bool)


class TestOpenTelemetryIntegration:
    """Test OpenTelemetry integration functions."""

    def test_init_opentelemetry_tracing_disabled_by_default(self, monkeypatch):
        """Test that OpenTelemetry tracing is disabled by default."""
        from llm_conductor.observability import init_opentelemetry_tracing

        result = init_opentelemetry_tracing()

        assert result is False

    @patch("llm_conductor.observability.OTEL_AVAILABLE", False)
    def test_init_opentelemetry_tracing_not_installed(self, monkeypatch):
        """Test OpenTelemetry tracing when dependencies not installed."""
        from llm_conductor.observability import init_opentelemetry_tracing

        monkeypatch.setenv("LITELLM_OTEL_ENABLED", "true")

        result = init_opentelemetry_tracing()

        assert result is False

    def test_otel_available_constant(self):
        """Test that OTEL_AVAILABLE constant is available."""
        from llm_conductor.observability import OTEL_AVAILABLE

        assert isinstance(OTEL_AVAILABLE, bool)


class TestObservabilityInit:
    """Test main init_observability function."""

    @patch("llm_conductor.observability.init_mlflow_tracing")
    @patch("llm_conductor.observability.init_opentelemetry_tracing")
    def test_init_observability_both_disabled(
        self, mock_otel_init, mock_mlflow_init, monkeypatch
    ):
        """Test init_observability when both backends disabled."""
        from llm_conductor.observability import init_observability

        mock_mlflow_init.return_value = False
        mock_otel_init.return_value = False

        result = init_observability()

        assert result is False
        mock_mlflow_init.assert_called_once()
        mock_otel_init.assert_called_once()

    @patch("llm_conductor.observability.init_mlflow_tracing")
    @patch("llm_conductor.observability.init_opentelemetry_tracing")
    def test_init_observability_mlflow_only(
        self, mock_otel_init, mock_mlflow_init, monkeypatch
    ):
        """Test init_observability with only MLflow enabled."""
        from llm_conductor.observability import init_observability

        mock_mlflow_init.return_value = True
        mock_otel_init.return_value = False

        result = init_observability()

        assert result is True
        mock_mlflow_init.assert_called_once()
        mock_otel_init.assert_called_once()

    @patch("llm_conductor.observability.init_mlflow_tracing")
    @patch("llm_conductor.observability.init_opentelemetry_tracing")
    def test_init_observability_otel_only(
        self, mock_otel_init, mock_mlflow_init, monkeypatch
    ):
        """Test init_observability with only OpenTelemetry enabled."""
        from llm_conductor.observability import init_observability

        mock_mlflow_init.return_value = False
        mock_otel_init.return_value = True

        result = init_observability()

        assert result is True
        mock_mlflow_init.assert_called_once()
        mock_otel_init.assert_called_once()

    @patch("llm_conductor.observability.init_mlflow_tracing")
    @patch("llm_conductor.observability.init_opentelemetry_tracing")
    def test_init_observability_both_enabled(
        self, mock_otel_init, mock_mlflow_init, monkeypatch
    ):
        """Test init_observability with both backends enabled."""
        from llm_conductor.observability import init_observability

        mock_mlflow_init.return_value = True
        mock_otel_init.return_value = True

        result = init_observability()

        assert result is True
        mock_mlflow_init.assert_called_once()
        mock_otel_init.assert_called_once()


class TestObservabilityConstants:
    """Test observability module constants."""

    def test_mlflow_available_constant_exists(self):
        """Test that MLFLOW_AVAILABLE constant exists."""
        from llm_conductor.mlflow_integration import MLFLOW_AVAILABLE

        assert isinstance(MLFLOW_AVAILABLE, bool)

    def test_otel_available_constant_exists(self):
        """Test that OTEL_AVAILABLE constant exists."""
        from llm_conductor.observability import OTEL_AVAILABLE

        assert isinstance(OTEL_AVAILABLE, bool)

    def test_mlflow_available_exported(self):
        """Test that MLFLOW_AVAILABLE is exported from observability module."""
        from llm_conductor.observability import MLFLOW_AVAILABLE

        assert isinstance(MLFLOW_AVAILABLE, bool)

    def test_observability_exports(self):
        """Test that observability module exports expected items."""
        from llm_conductor import observability

        assert hasattr(observability, "MLFLOW_AVAILABLE")
        assert hasattr(observability, "OTEL_AVAILABLE")
        assert hasattr(observability, "init_observability")
        assert hasattr(observability, "init_opentelemetry_tracing")
