"""OpenAI Codex CLI provider."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import override

from llm_conductor.cli_base import CLIProvider


class OpenAICodexProvider(CLIProvider):
    """Provider for OpenAI Codex CLI (codex command)."""

    def __init__(
        self, model_name: str | None = None, reasoning_effort: str | None = None
    ) -> None:
        super().__init__(model_name)
        self.reasoning_effort = reasoning_effort or os.getenv(
            "CODEX_REASONING_EFFORT", "medium"
        )

    @override
    def _get_encoding_name(self) -> str:
        """Get tiktoken encoding for OpenAI models.

        Returns
        -------
            cl100k_base encoding for GPT-4/5 models

        """
        return "cl100k_base"

    @override
    def run(self, prompt: str) -> str:
        """Run Codex CLI with the prompt."""
        try:
            # Check if codex CLI is available
            result = subprocess.run(
                ["which", "codex"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                msg = "codex CLI not found. Install from OpenAI Codex documentation"
                raise RuntimeError(msg)

            # Create temporary file for output
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".txt", delete=False
            ) as tmp_output:
                output_file = tmp_output.name

            try:
                # Run codex exec with stdin
                cmd = [
                    "codex",
                    "exec",
                    "-m",
                    self.model_name or "gpt-5",
                    "--sandbox",
                    "workspace-write",
                    "-c",
                    f"model_reasoning_effort={self.reasoning_effort}",
                    "--output-last-message",
                    output_file,
                ]
                result = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                # Check for errors in stderr or stdout (codex prints errors there)
                combined_output = result.stdout + result.stderr
                if result.returncode != 0:
                    msg = f"codex CLI failed with exit code {result.returncode}: {result.stderr}"
                    raise RuntimeError(msg)

                # Check for usage limit errors (codex exits with 0 but prints error)
                if (
                    "usage limit" in combined_output.lower()
                    or "ERROR:" in combined_output
                ):
                    msg = f"codex CLI error: {combined_output}"
                    raise RuntimeError(msg)

                # Read output from file
                with Path(output_file).open() as f:
                    content = f.read().strip()

                # Validate output isn't empty
                if not content:
                    msg = f"codex CLI returned empty output. stderr: {result.stderr}"
                    raise RuntimeError(msg)

                return content

            finally:
                # Cleanup temp file
                output_path = Path(output_file)
                if output_path.exists():
                    output_path.unlink()

        except FileNotFoundError as e:
            msg = f"Failed to execute codex: {e}"
            raise RuntimeError(msg) from e
