"""Claude Code CLI provider."""

from __future__ import annotations

import subprocess
from typing import override

from llm_conductor.cli_base import CLIProvider


class ClaudeCodeProvider(CLIProvider):
    """Provider for Claude Code CLI (claude command)."""

    @override
    def _get_encoding_name(self) -> str:
        """Get tiktoken encoding for Claude models.

        Returns
        -------
            cl100k_base encoding (GPT-4) as approximation for Claude

        """
        return "cl100k_base"

    @override
    def run(self, prompt: str) -> str:
        """Run Claude CLI with the prompt."""
        try:
            # Check if claude CLI is available
            result = subprocess.run(
                ["which", "claude"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                msg = "claude CLI not found. Install with: npm install -g @anthropic-ai/claude-cli"
                raise RuntimeError(msg)

            # Build command with optional model specification
            cmd = ["claude", "-p", prompt]
            if self.model_name:
                cmd.extend(["--model", self.model_name])

            # Run claude with -p flag for prompt
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                msg = f"claude CLI failed with exit code {result.returncode}: {result.stderr}"
                raise RuntimeError(msg)

            return result.stdout

        except FileNotFoundError as e:
            msg = f"Failed to execute claude: {e}"
            raise RuntimeError(msg) from e
