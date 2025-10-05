"""Unit tests for CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from llm_conductor.cli import main


class TestListCommand:
    """Test the 'list' command."""

    def test_list_command_success(self):
        """Test that list command shows all providers."""
        runner = CliRunner()
        result = runner.invoke(main, ["list"])

        assert result.exit_code == 0
        assert "Available LLM Providers:" in result.output
        assert "claude-code" in result.output
        assert "openai-codex" in result.output
        assert "openai" in result.output
        assert "lmstudio" in result.output
        assert "openrouter" in result.output


class TestRunCommand:
    """Test the 'run' command."""

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_single_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_run_command_basic(
        self, mock_observability, mock_run_single_mode, mock_get_provider
    ):
        """Test basic run command invocation."""
        # Setup mocks
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_single_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--provider", "openai"])

        # run command calls sys.exit(), so exit_code comes from that
        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("openai", None, None)
        mock_run_single_mode.assert_called_once_with(mock_provider, False)
        mock_observability.assert_called()

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_single_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_run_command_with_model(
        self, mock_observability, mock_run_single_mode, mock_get_provider
    ):
        """Test run command with explicit model."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_single_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(
            main, ["run", "--provider", "openai", "--model", "gpt-4o"]
        )

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("openai", "gpt-4o", None)

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_single_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_run_command_with_reasoning(
        self, mock_observability, mock_run_single_mode, mock_get_provider
    ):
        """Test run command with reasoning parameter."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_single_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(
            main, ["run", "--provider", "openai-codex", "--reasoning", "low"]
        )

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("openai-codex", None, "low")

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_single_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_run_command_verbose(
        self, mock_observability, mock_run_single_mode, mock_get_provider
    ):
        """Test run command with verbose flag."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_single_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--provider", "openai", "--verbose"])

        assert result.exit_code == 0
        mock_run_single_mode.assert_called_once_with(mock_provider, True)

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.init_observability")
    def test_run_command_provider_error(self, mock_observability, mock_get_provider):
        """Test run command handles provider creation error."""
        mock_get_provider.side_effect = ValueError("Unknown provider")

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--provider", "invalid"])

        assert result.exit_code == 1
        assert "Run failed" in result.output

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_single_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_run_command_env_vars(
        self, mock_observability, mock_run_single_mode, mock_get_provider, monkeypatch
    ):
        """Test run command respects environment variables."""
        monkeypatch.setenv("MODEL_PROVIDER", "openrouter")
        monkeypatch.setenv("MODEL_NAME", "test-model")
        monkeypatch.setenv("MODEL_REASONING", "medium")

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_single_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(main, ["run"])

        assert result.exit_code == 0
        # Should use env vars
        mock_get_provider.assert_called_once_with("openrouter", "test-model", "medium")


class TestBatchCommand:
    """Test the 'batch' command."""

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_batch_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_batch_command_basic(
        self,
        mock_observability,
        mock_run_batch_mode,
        mock_get_provider,
        tmp_path,
    ):
        """Test basic batch command invocation."""
        # Create test directories and prompt file
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        input_dir.mkdir()

        prompt_file = input_dir / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        # Setup mocks
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_batch_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                "--provider",
                "openai",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        mock_get_provider.assert_called_once_with("openai", None, "high")
        mock_run_batch_mode.assert_called_once()

        # Check that provider was passed and function was called with correct types
        call_args = mock_run_batch_mode.call_args
        assert call_args[0][0] == mock_provider  # provider
        assert isinstance(call_args[0][1], list)  # prompt_files list
        assert isinstance(call_args[0][2], int)  # parallelism
        assert isinstance(call_args[0][3], bool)  # verbose
        assert isinstance(call_args[0][4], bool)  # force

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_batch_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_batch_command_parallelism(
        self, mock_observability, mock_run_batch_mode, mock_get_provider, tmp_path
    ):
        """Test batch command with custom parallelism."""
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        input_dir.mkdir()

        prompt_file = input_dir / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_batch_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                "--provider",
                "openai",
                "--parallelism",
                "10",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        call_args = mock_run_batch_mode.call_args
        assert call_args[0][2] == 10  # parallelism

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_batch_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_batch_command_lmstudio_default_parallelism(
        self, mock_observability, mock_run_batch_mode, mock_get_provider, tmp_path
    ):
        """Test that lmstudio provider defaults to parallelism=1."""
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        input_dir.mkdir()

        prompt_file = input_dir / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_batch_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                "--provider",
                "lmstudio",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        call_args = mock_run_batch_mode.call_args
        assert call_args[0][2] == 1  # parallelism for lmstudio

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_batch_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_batch_command_cli_provider_default_parallelism(
        self, mock_observability, mock_run_batch_mode, mock_get_provider, tmp_path
    ):
        """Test that CLI providers default to parallelism=4."""
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        input_dir.mkdir()

        prompt_file = input_dir / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_batch_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                "--provider",
                "claude-code",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        call_args = mock_run_batch_mode.call_args
        assert call_args[0][2] == 4  # parallelism for CLI providers

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_batch_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_batch_command_recursive(
        self,
        mock_observability,
        mock_run_batch_mode,
        mock_get_provider,
        tmp_path,
    ):
        """Test batch command with recursive flag."""
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        input_dir.mkdir()

        # Create nested structure
        subdir = input_dir / "subdir"
        subdir.mkdir()
        (input_dir / "test1.prompt.md").write_text("Test 1")
        (subdir / "test2.prompt.md").write_text("Test 2")

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_batch_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                "--provider",
                "openai",
                "--recursive",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        # Verify batch mode was called with some files
        call_args = mock_run_batch_mode.call_args
        prompt_files = call_args[0][1]
        assert isinstance(prompt_files, list)

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_batch_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_batch_command_force_flag(
        self, mock_observability, mock_run_batch_mode, mock_get_provider, tmp_path
    ):
        """Test batch command with force flag."""
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        input_dir.mkdir()

        prompt_file = input_dir / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_batch_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                "--provider",
                "openai",
                "--force",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        call_args = mock_run_batch_mode.call_args
        assert call_args[0][4] is True  # force flag

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_batch_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_batch_command_verbose_flag(
        self, mock_observability, mock_run_batch_mode, mock_get_provider, tmp_path
    ):
        """Test batch command with verbose flag."""
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        input_dir.mkdir()

        prompt_file = input_dir / "test.prompt.md"
        prompt_file.write_text("Test prompt")

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_batch_mode.return_value = 0

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                "--provider",
                "openai",
                "--verbose",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        call_args = mock_run_batch_mode.call_args
        assert call_args[0][3] is True  # verbose flag

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.init_observability")
    def test_batch_command_missing_input_dir(
        self, mock_observability, mock_get_provider
    ):
        """Test batch command with missing input directory."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                "--provider",
                "openai",
                "--input-dir",
                "/nonexistent",
                "--output-dir",
                "/tmp/out",
            ],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    @patch("llm_conductor.cli.get_provider")
    @patch("llm_conductor.cli.run_batch_mode")
    @patch("llm_conductor.cli.init_observability")
    def test_batch_command_provider_error(
        self, mock_observability, mock_run_batch_mode, mock_get_provider, tmp_path
    ):
        """Test batch command handles provider creation error."""
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        input_dir.mkdir()

        mock_get_provider.side_effect = ValueError("Unknown provider")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                "--provider",
                "invalid",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 1
        assert "Batch run failed" in result.output


class TestMainGroup:
    """Test the main CLI group."""

    def test_main_help(self):
        """Test main help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "LLM Conductor" in result.output
        assert "run" in result.output
        assert "batch" in result.output
        assert "list" in result.output

    def test_main_version(self):
        """Test version flag."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        # Version output format may vary, but should exit cleanly
