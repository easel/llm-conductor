"""Model provider execution logic for batch and single-prompt modes."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from json_repair import repair_json

if TYPE_CHECKING:
    from llm_conductor.base import ModelProvider


# Task log formatting functions for consistent message structure
def format_task_status(
    emoji: str,
    task_num: int,
    total_tasks: int,
    filename: str,
    details: str | None = None,
    error: str | None = None,
) -> str:
    """Format task status with consistent structure.

    Format: {emoji} [{task_num}/{total_tasks}] {filename} ({details})
    Error format: {emoji} [{task_num}/{total_tasks}] {filename}: {error} ({details})

    Args:
    ----
        emoji: Status emoji (⏭️, 🚀, ✅, ❌, ⏳)
        task_num: Current task number
        total_tasks: Total number of tasks
        filename: Output filename
        details: Optional details in parentheses
        error: Optional error message (for error format)

    Returns:
    -------
        Formatted log message

    """
    base = f"{emoji} [{task_num}/{total_tasks}] {filename}"
    if error:
        return f"{base}: {error} ({details})" if details else f"{base}: {error}"
    if details:
        return f"{base} ({details})"
    return base


def format_start_details(model: str, prompt_length: int, max_tokens: str | int) -> str:
    """Format task start details.

    Args:
    ----
        model: Model name
        prompt_length: Prompt length in characters
        max_tokens: Maximum tokens setting

    Returns:
    -------
        Formatted details string

    """
    return f"{model}, prompt: {prompt_length} chars, max_tokens: {max_tokens}"


def format_skip_details(reason: str) -> str:
    """Format skip reason.

    Args:
    ----
        reason: Skip reason ('unchanged' or 'locked')

    Returns:
    -------
        Formatted details string

    """
    return reason


def format_success_details(elapsed: float, usage: dict | None = None) -> str:
    """Format success details with optional token metrics.

    Args:
    ----
        elapsed: Elapsed time in seconds
        usage: Optional usage dict with token counts

    Returns:
    -------
        Formatted details string

    """
    details = f"{elapsed:.1f}s"
    if usage and usage.get("total_tokens"):
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage["total_tokens"]
        tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        details += f", {total_tokens} tok ({prompt_tokens}→{completion_tokens}), {tok_per_sec:.0f} tok/s"
    return details


def format_streaming_details(chunk_count: int, chars: int, elapsed: float) -> str:
    """Format streaming progress details.

    Args:
    ----
        chunk_count: Number of chunks received
        chars: Character count so far
        elapsed: Elapsed time in seconds

    Returns:
    -------
        Formatted details string

    """
    return f"streaming: {chunk_count} chunks, ~{chars} chars, {elapsed:.1f}s elapsed"


def setup_logging() -> logging.Logger:
    """Configure logging with both console and file handlers.

    Returns
    -------
        Configured logger instance

    """
    log_dir = Path(".local/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "llm-conductor.log"

    # Get or create logger for llm-conductor
    logger = logging.getLogger("llm_conductor")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler - shows INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler - captures DEBUG and above
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def _extract_json_hash(path: Path) -> str | None:
    """Extract hash from JSON file."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        return data.get("_prompt_hash") if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_md_hash(path: Path) -> str | None:
    """Extract hash from markdown file."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            for _ in range(5):
                line = f.readline()
                if line.startswith("<!-- Prompt Hash: "):
                    return (
                        line.replace("<!-- Prompt Hash: ", "")
                        .replace(" -->", "")
                        .strip()
                    )
        return None
    except Exception:
        return None


def extract_existing_hash(output_path: str) -> str | None:
    """Extract hash from existing output file if it exists.

    Handles both base paths (without extension) and full paths (with extension).

    Args:
    ----
        output_path: Base path or full path to the output file

    Returns:
    -------
        Hash string if found, None otherwise

    """
    path = Path(output_path)

    # If path already has .json or .md extension, use it directly
    if path.suffix == ".json":
        return _extract_json_hash(path)
    if path.suffix == ".md":
        return _extract_md_hash(path)

    # No extension - check for both .json and .md
    json_hash = _extract_json_hash(Path(str(path) + ".json"))
    if json_hash:
        return json_hash

    return _extract_md_hash(Path(str(path) + ".md"))


def compute_prompt_hash(prompt_path: Path) -> str:
    """Compute SHA-256 hash of prompt file contents.

    Args:
    ----
        prompt_path: Path to prompt file

    Returns:
    -------
        SHA-256 hash as hex string

    """
    with Path(prompt_path).open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_output_path(input_path: Path, input_dir: Path, output_dir: Path) -> str:
    """Compute output path, preserving embedded extension hint if present.

    Supports two patterns:
    - {name}.{ext}.prompt.md → {name}.{ext} (with extension hint)
    - {name}.prompt.md → {name} (no hint, for backward compatibility)

    Args:
    ----
        input_path: Absolute path to input prompt file
        input_dir: Base input directory
        output_dir: Base output directory

    Returns:
    -------
        Absolute output path (may include extension hint like .json or .md)

    """
    # Get relative path from input_dir
    rel_path = input_path.relative_to(input_dir)

    # Strip .prompt.md extension, preserving any embedded extension
    # Example: "foo.json.prompt.md" → "foo.json"
    name = rel_path.name.removesuffix(".prompt.md")

    rel_path = rel_path.with_name(name)

    # Return path with embedded extension preserved
    return str(output_dir / rel_path)


def acquire_lock(output_path: str, provider_name: str = "unknown") -> bool:
    """Acquire a lock file for the given output path.

    Creates a lock file with metadata to prevent concurrent processing.
    Handles stale locks (>5 minutes old) by taking over.

    Args:
    ----
        output_path: Base output path (without extension)
        provider_name: Name of the provider acquiring the lock

    Returns:
    -------
        True if lock acquired, False if locked by another process

    """
    import os
    import socket

    lock_path = f"{output_path}.lock"

    # Check if lock exists
    if Path(lock_path).exists():
        try:
            with Path(lock_path).open() as f:
                lock_data = json.load(f)

            lock_age = time.time() - lock_data.get("timestamp", 0)

            # If lock is less than 5 minutes old, respect it
            if lock_age < 300:  # 5 minutes
                return False

            # Stale lock - log and take over
            logger = logging.getLogger("llm_conductor")
            logger.warning(
                f"⚠️  Taking over stale lock (age: {lock_age:.0f}s, "
                f"pid: {lock_data.get('pid')}, host: {lock_data.get('hostname')})"
            )
        except Exception:
            # Corrupted lock file - take over
            pass

    # Create lock file
    lock_data = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "timestamp": time.time(),
        "provider": provider_name,
    }

    try:
        with Path(lock_path).open("w") as f:
            json.dump(lock_data, f)
        return True
    except Exception:
        return False


def release_lock(output_path: str) -> None:
    """Release the lock file for the given output path.

    Args:
    ----
        output_path: Base output path (without extension)

    """
    lock_path = f"{output_path}.lock"
    with contextlib.suppress(Exception):
        Path(lock_path).unlink(missing_ok=True)


def strip_markdown_code_blocks(content: str) -> str:
    r"""Strip markdown wrappers and extract JSON content.

    Aggressive stripping rules for JSON extraction:
    - If ≤3 lines before ```, strip everything before first {/[
    - Strip everything after last }/]
    - Handle both fenced code blocks and inline JSON

    Args:
    ----
        content: The raw content from model

    Returns:
    -------
        Content with markdown wrappers removed

    """
    content = content.strip()

    # Find first valid JSON character ({ or [)
    brace_pos = content.find("{")
    bracket_pos = content.find("[")

    if brace_pos == -1 and bracket_pos == -1:
        # No JSON found, return as-is
        return content

    first_json_pos = min(p for p in [brace_pos, bracket_pos] if p != -1)

    # Check for ``` fence before JSON
    fence_pos = content.find("```")
    should_strip = False

    if fence_pos != -1 and fence_pos < first_json_pos:
        # Count newlines before fence
        lines_before = content[:fence_pos].count("\n")
        should_strip = lines_before <= 3

    # If preamble detected or fence exists, strip to first JSON char
    if should_strip or fence_pos != -1:
        content = content[first_json_pos:]

    # Find last valid JSON character (} or ])
    last_brace = content.rfind("}")
    last_bracket = content.rfind("]")

    if last_brace == -1 and last_bracket == -1:
        return content

    last_json_pos = max(p for p in [last_brace, last_bracket] if p != -1)

    # Strip everything after last JSON char
    return content[: last_json_pos + 1]


def is_json_content(content: str) -> bool:
    """Check if content is valid JSON (object or array).

    Args:
    ----
        content: Stripped content to test

    Returns:
    -------
        True if valid JSON dict/list, False otherwise

    """
    try:
        parsed = json.loads(content)
        return isinstance(parsed, dict | list)
    except (json.JSONDecodeError, ValueError):
        return False


def write_output_file(
    content: str,
    base_output_path: str,
    prompt_hash: str,
    llm_metadata: dict | None = None,
) -> tuple[str, str]:
    """Write output file with explicit format handling based on extension hint.

    Supports three modes:
    1. Explicit .json: Strip markdown, parse JSON, fail if invalid
    2. Explicit .md: Write as markdown
    3. No extension (backward compat): Auto-detect format

    Args:
    ----
        content: Raw content from model
        base_output_path: Output path (may include .json or .md hint)
        prompt_hash: Hash to embed in output
        llm_metadata: Optional LLM metadata to embed (provider, model, tokens, timing)

    Returns:
    -------
        Tuple of (final_output_path, extension_used)

    """
    # Check for explicit format hint in base_output_path
    path_obj = Path(base_output_path)
    has_json_hint = path_obj.suffix == ".json"
    has_md_hint = path_obj.suffix == ".md"

    # Mode 1: Explicit JSON output expected
    if has_json_hint:
        final_path = base_output_path
        stripped = strip_markdown_code_blocks(content)

        try:
            data = json.loads(stripped)
            # Embed hash and metadata if dict, otherwise write as-is
            if isinstance(data, dict):
                data["_prompt_hash"] = prompt_hash
                if llm_metadata:
                    data["_llm_metadata"] = llm_metadata
                Path(final_path).parent.mkdir(parents=True, exist_ok=True)
                with Path(final_path).open("w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                # JSON array - write as-is (can't embed hash or metadata)
                Path(final_path).parent.mkdir(parents=True, exist_ok=True)
                with Path(final_path).open("w") as f:
                    f.write(stripped)
            return (final_path, ".json")
        except json.JSONDecodeError as e:
            # Try to repair JSON before failing
            logger = logging.getLogger("llm_conductor")
            logger.warning(
                f"JSON parse failed for {Path(final_path).name}: {e}. Attempting repair..."
            )
            try:
                repaired = repair_json(stripped)
                data = json.loads(repaired)
                logger.info(f"✓ JSON repair successful for {Path(final_path).name}")

                # Embed hash and metadata if dict, otherwise write as-is
                if isinstance(data, dict):
                    data["_prompt_hash"] = prompt_hash
                    if llm_metadata:
                        data["_llm_metadata"] = llm_metadata
                    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
                    with Path(final_path).open("w") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    # JSON array - write as-is (can't embed hash or metadata)
                    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
                    with Path(final_path).open("w") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                return (final_path, ".json")
            except (json.JSONDecodeError, Exception) as repair_error:
                # Repair failed - raise original error
                logger.exception(
                    f"✗ JSON repair failed for {Path(final_path).name}: {repair_error}"
                )
                msg = f"Expected JSON output but content is not valid JSON: {e}"
                raise ValueError(msg) from e

    # Mode 2: Explicit markdown output expected
    if has_md_hint:
        final_path = base_output_path
        Path(final_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(final_path).open("w") as f:
            # Write metadata as HTML comments
            f.write(f"<!-- Prompt Hash: {prompt_hash} -->\n")
            if llm_metadata:
                f.write(f"<!-- LLM Metadata: {json.dumps(llm_metadata)} -->\n")
            f.write(f"\n{content}")
        return (final_path, ".md")

    # Mode 3: No hint - auto-detect (backward compatibility)
    stripped = strip_markdown_code_blocks(content)
    is_json = is_json_content(stripped)

    ext = ".json" if is_json else ".md"
    final_path = base_output_path + ext
    content_to_write = stripped if is_json else content

    Path(final_path).parent.mkdir(parents=True, exist_ok=True)

    if is_json:
        try:
            data = json.loads(content_to_write)
            if isinstance(data, dict):
                data["_prompt_hash"] = prompt_hash
                if llm_metadata:
                    data["_llm_metadata"] = llm_metadata
                with Path(final_path).open("w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with Path(final_path).open("w") as f:
                    f.write(content_to_write)
        except json.JSONDecodeError:
            # Fallback: write raw
            with Path(final_path).open("w") as f:
                f.write(content_to_write)
    else:
        with Path(final_path).open("w") as f:
            f.write(f"<!-- Prompt Hash: {prompt_hash} -->\n")
            if llm_metadata:
                f.write(f"<!-- LLM Metadata: {json.dumps(llm_metadata)} -->\n")
            f.write(f"\n{content_to_write}")

    return (final_path, ext)


def process_single_task(
    provider: ModelProvider,
    prompt_path: str,
    output_path: str,
    prompt_hash: str,
) -> tuple[str, bool, str | None, float, dict | None, str | None]:
    """Process a single prompt file and write output (sync version).

    Args:
    ----
        provider: Model provider instance
        prompt_path: Path to prompt file
        output_path: Path to output file
        prompt_hash: Hash to embed in output

    Returns:
    -------
        Tuple of (output_filename, success, error_message, elapsed_time, usage, generation_id)
        Note: CLI providers return None for usage and generation_id

    """
    start_time = time.time()

    try:
        # Read prompt from file
        with Path(prompt_path).open() as f:
            prompt = f.read()

        # Run model (use run_with_usage if available for token counting)
        usage = None
        if hasattr(provider, "run_with_usage"):
            content, usage = provider.run_with_usage(prompt)
        else:
            content = provider.run(prompt)

        elapsed = time.time() - start_time

        # Collect metadata
        provider_info = provider.get_provider_info()
        llm_metadata = {
            "provider": provider_info["provider"],
            "model": provider_info["model"],
            "temperature": provider_info["temperature"],
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "completion_tokens": usage.get("completion_tokens") if usage else None,
            "total_tokens": usage.get("total_tokens") if usage else None,
            "elapsed_seconds": round(elapsed, 2),
            "generation_date": datetime.now(UTC).isoformat(),
            "generation_id": None,  # CLI providers don't provide generation IDs
            "cost_usd": usage.get("cost_usd") or usage.get("estimated_cost_usd")
            if usage
            else None,
            "cost_calculation_method": usage.get("cost_calculation_method")
            if usage
            else None,
        }

        # Unified write (handles stripping, detection, extension, formatting)
        final_path, _ext = write_output_file(
            content, output_path, prompt_hash, llm_metadata
        )

        output_name = Path(final_path).name
        # CLI providers don't provide usage or generation_id
        return (output_name, True, None, elapsed, None, None)

    except Exception as e:
        elapsed = time.time() - start_time
        # Use base output_path for error reporting
        output_name = Path(output_path).name
        return (output_name, False, str(e), elapsed, None, None)


async def async_process_single_task(
    provider: ModelProvider,
    prompt_path: str,
    output_path: str,
    prompt_hash: str,
) -> tuple[str, bool, str | None, float, dict | None, str | None]:
    """Process a single prompt file and write output (async version).

    Args:
    ----
        provider: Model provider instance (must have async_run method)
        prompt_path: Path to prompt file
        output_path: Path to output file
        prompt_hash: Hash to embed in output

    Returns:
    -------
        Tuple of (output_filename, success, error_message, elapsed_time, usage, generation_id)

    """
    start_time = time.time()

    try:
        # Read prompt from file
        with Path(prompt_path).open() as f:  # noqa: ASYNC230
            prompt = f.read()

        # Run model asynchronously
        result = await provider.async_run(prompt)  # type: ignore[attr-defined]

        # Handle tuple return (content, usage, generation_id) from LiteLLM providers
        if isinstance(result, tuple):
            content, usage, generation_id = result
        else:
            # Fallback for providers that return just string (shouldn't happen for async)
            content = result
            usage = None
            generation_id = None

        elapsed = time.time() - start_time

        # Collect metadata
        provider_info = provider.get_provider_info()
        llm_metadata = {
            "provider": provider_info["provider"],
            "model": provider_info["model"],
            "temperature": provider_info["temperature"],
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "completion_tokens": usage.get("completion_tokens") if usage else None,
            "total_tokens": usage.get("total_tokens") if usage else None,
            "elapsed_seconds": round(elapsed, 2),
            "generation_date": datetime.now(UTC).isoformat(),
            "generation_id": generation_id,
            "cost_usd": usage.get("cost_usd") or usage.get("estimated_cost_usd")
            if usage
            else None,
            "cost_calculation_method": usage.get("cost_calculation_method")
            if usage
            else None,
        }

        # Unified write (handles stripping, detection, extension, formatting)
        final_path, _ext = write_output_file(
            content, output_path, prompt_hash, llm_metadata
        )

        output_name = Path(final_path).name
        return (output_name, True, None, elapsed, usage, generation_id)

    except Exception as e:
        elapsed = time.time() - start_time
        # Use base output_path for error reporting
        output_name = Path(output_path).name
        return (output_name, False, str(e), elapsed, None, None)


def run_batch_mode(
    provider: ModelProvider,
    prompt_files: list[Path],
    parallelism: int,
    verbose: bool,
    force: bool = False,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> int:
    """Run in batch mode, processing a list of prompt files.

    Args:
    ----
        provider: Model provider instance
        prompt_files: List of prompt file paths to process
        parallelism: Number of parallel workers
        verbose: Enable verbose logging
        force: Force regeneration even if hash matches
        input_dir: Base directory for input prompts (required)
        output_dir: Base directory for output files (required)

    Returns:
    -------
        Exit code (0 for success)

    """
    # Set up logging
    logger = setup_logging()

    # Validate required directories
    if not input_dir or not output_dir:
        logger.error("Error: --input-dir and --output-dir are required")
        return 1

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    # Resolve prompt_files to absolute paths (handles both relative and absolute)
    prompt_files = [p.resolve() for p in prompt_files]

    if not input_dir.exists():
        logger.error(f"Error: Input directory does not exist: {input_dir}")
        return 1

    if not prompt_files:
        logger.error("Error: No prompt files provided")
        return 1

    # Build task list from provided prompt files
    task_list = []
    for prompt_file in prompt_files:
        if not prompt_file.exists():
            logger.warning(f"⚠️  [SKIP] File not found: {prompt_file}")
            continue

        output_path = compute_output_path(prompt_file, input_dir, output_dir)
        prompt_hash = compute_prompt_hash(prompt_file)
        task_list.append((str(prompt_file), str(output_path), prompt_hash))

    if not task_list:
        logger.error("Error: No valid tasks to process")
        return 1

    logger.info(f"✓ Processing {len(task_list)} prompt files")

    total_tasks = len(task_list)
    tasks_to_process = task_list

    # Display processing summary
    logger.info(
        f"📋 Processing up to {len(tasks_to_process)} tasks with {parallelism} workers"
    )
    if verbose:
        logger.info(f"   Provider: {provider.__class__.__name__}")
        logger.info(f"   Model: {provider.model_name}")
        if force:
            logger.info("   Force mode: regenerating all outputs")

    # Log batch execution details
    logger.info(
        f"Batch mode: {len(tasks_to_process)} tasks, {parallelism} workers, "
        f"provider={provider.__class__.__name__}, model={provider.model_name}, force={force}"
    )

    # Process tasks in parallel - use asyncio for async providers, ThreadPoolExecutor for sync
    completed = 0
    failed = 0
    skipped = 0
    has_async = hasattr(provider, "async_run")

    if has_async:
        # Use asyncio for async providers (LiteLLM-based) with bounded concurrency
        async def run_all_async() -> None:
            nonlocal completed, failed, skipped
            # Semaphore limits concurrent tasks to parallelism count
            semaphore = asyncio.Semaphore(parallelism)
            heartbeat_running = True
            # Track last INFO log time for heartbeat suppression
            last_info_log_time = [asyncio.get_event_loop().time()]

            async def bounded_task(
                task_num: int, prompt_path: str, output_path: str, prompt_hash: str
            ) -> tuple[str, bool, str | None, float, dict | None, str | None, bool]:
                # Check hash before acquiring semaphore (allows other processes to collaborate)
                if not force:
                    existing_hash = extract_existing_hash(output_path)
                    if existing_hash == prompt_hash:
                        output_name = Path(output_path).name
                        # Log skip with [n/m] at INFO level
                        logger.info(
                            format_task_status(
                                "⏭️ ",
                                task_num,
                                len(tasks_to_process),
                                output_name,
                                format_skip_details("unchanged"),
                            )
                        )
                        last_info_log_time[0] = asyncio.get_event_loop().time()
                        return (output_name, True, None, 0.0, None, None, True)

                # Check if file is locked by another process (before acquiring semaphore)
                if not acquire_lock(output_path, provider.__class__.__name__):
                    output_name = Path(output_path).name
                    # Log locked skip with [n/m] at INFO level
                    logger.info(
                        format_task_status(
                            "⏭️ ",
                            task_num,
                            len(tasks_to_process),
                            output_name,
                            format_skip_details("locked"),
                        )
                    )
                    last_info_log_time[0] = asyncio.get_event_loop().time()
                    return (output_name, True, None, 0.0, None, None, True)

                try:
                    async with semaphore:
                        # Log start with full context
                        output_name = Path(output_path).name
                        prompt_length = Path(prompt_path).stat().st_size
                        provider_info = provider.get_provider_info()
                        max_tokens = getattr(provider, "max_tokens", "unknown")
                        logger.info(
                            format_task_status(
                                "🚀",
                                task_num,
                                len(tasks_to_process),
                                output_name,
                                format_start_details(
                                    provider_info["model"], prompt_length, max_tokens
                                ),
                            )
                        )
                        last_info_log_time[0] = asyncio.get_event_loop().time()

                        # Install logging filter to add [n/m] context to provider logs (if provider has logger)
                        task_filter = None
                        if hasattr(provider, "logger"):

                            class TaskContextFilter(logging.Filter):
                                def filter(self, record):
                                    msg = record.getMessage()
                                    # Intercept streaming progress messages
                                    if "⏳ Streaming" in msg:
                                        # Parse: "⏳ Streaming... X chunks, ~Y chars, Z.Zs elapsed"
                                        try:
                                            parts = msg.split("... ")[
                                                1
                                            ]  # Get "X chunks, ~Y chars, Z.Zs elapsed"
                                            chunk_count = int(parts.split(" chunks")[0])
                                            chars = int(
                                                parts.split("~")[1].split(" chars")[0]
                                            )
                                            elapsed = float(
                                                parts.split(", ")[2].split("s elapsed")[
                                                    0
                                                ]
                                            )
                                            details = format_streaming_details(
                                                chunk_count, chars, elapsed
                                            )
                                            record.msg = format_task_status(
                                                "⏳",
                                                task_num,
                                                len(tasks_to_process),
                                                output_name,
                                                details,
                                            )
                                        except (IndexError, ValueError):
                                            # Fallback if parsing fails
                                            record.msg = f"⏳ [{task_num}/{len(tasks_to_process)}] {output_name} - {msg}"
                                        # Update last log time for these progress messages
                                        last_info_log_time[0] = (
                                            asyncio.get_event_loop().time()
                                        )
                                    elif "🔄 Response truncated" in msg:
                                        record.msg = f"🔄 [{task_num}/{len(tasks_to_process)}] {output_name} - {msg}"
                                        last_info_log_time[0] = (
                                            asyncio.get_event_loop().time()
                                        )
                                    return True

                            task_filter = TaskContextFilter()
                            provider.logger.addFilter(task_filter)

                        try:
                            result = await async_process_single_task(
                                provider, prompt_path, output_path, prompt_hash
                            )
                            # Add skipped=False to result tuple
                            return (*result, False)
                        finally:
                            if task_filter and hasattr(provider, "logger"):
                                provider.logger.removeFilter(task_filter)
                finally:
                    # Always release lock, even if processing fails
                    release_lock(output_path)

            async def heartbeat_task() -> None:
                """Print progress only when no INFO logs for >10s."""
                await asyncio.sleep(15)  # Initial delay
                while heartbeat_running:
                    # Only log if no other INFO logs in last 10 seconds
                    time_since_last_log = (
                        asyncio.get_event_loop().time() - last_info_log_time[0]
                    )

                    if time_since_last_log >= 10.0:
                        in_flight = len(tasks_to_process) - completed - failed - skipped
                        if in_flight > 0:
                            logger.info(
                                f"⏳ {in_flight} tasks in flight, {completed} completed, {failed} failed, {skipped} skipped..."
                            )
                            last_info_log_time[0] = asyncio.get_event_loop().time()

                    await asyncio.sleep(15)

            tasks = [
                asyncio.create_task(
                    bounded_task(idx + 1, prompt_path, output_path, prompt_hash)
                )
                for idx, (prompt_path, output_path, prompt_hash) in enumerate(
                    tasks_to_process
                )
            ]

            # Start heartbeat task
            heartbeat = asyncio.create_task(heartbeat_task())

            # Show submission complete message
            logger.info(
                f"🚀 Submitted {len(tasks)} async tasks, waiting for completions..."
            )

            # Process results as they complete
            for task in asyncio.as_completed(tasks):
                (
                    output_name,
                    success,
                    error,
                    elapsed,
                    usage,
                    generation_id,
                    is_skipped,
                ) = await task

                if is_skipped:
                    skipped += 1
                    # Skip already logged in bounded_task
                elif success:
                    completed += 1

                    # Format log message with optional tokens and tok/s
                    log_msg = format_task_status(
                        "✅",
                        completed + skipped,
                        len(tasks_to_process),
                        output_name,
                        format_success_details(elapsed, usage),
                    )
                    debug_msg = f"Task completed: {output_name} ({format_success_details(elapsed, usage)}"
                    if generation_id:
                        debug_msg += f", gen_id={generation_id}"
                    debug_msg += ")"

                    logger.info(log_msg)
                    logger.debug(debug_msg)
                    last_info_log_time[0] = asyncio.get_event_loop().time()
                else:
                    failed += 1
                    logger.error(
                        format_task_status(
                            "❌",
                            completed + failed + skipped,
                            len(tasks_to_process),
                            output_name,
                            f"{elapsed:.1f}s",
                            error,
                        )
                    )
                    logger.debug(
                        f"Task failed: {output_name} - {error} ({elapsed:.1f}s)"
                    )
                    last_info_log_time[0] = asyncio.get_event_loop().time()

            # Stop heartbeat task
            heartbeat_running = False
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

        asyncio.run(run_all_async())
    else:
        # Use ThreadPoolExecutor for sync providers (subprocess-based)
        # Track last INFO log time for consistent behavior with async path
        last_info_log_time = [time.time()]

        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            # Check hash and submit only tasks that need processing
            future_to_task = {}
            task_num_counter = [0]  # Track task submission order
            for prompt_path, output_path, prompt_hash in tasks_to_process:
                # Check hash just-in-time (allows other processes to collaborate)
                if not force:
                    existing_hash = extract_existing_hash(output_path)
                    if existing_hash == prompt_hash:
                        output_name = Path(output_path).name
                        skip_num = skipped + completed + failed + 1
                        logger.info(
                            format_task_status(
                                "⏭️ ",
                                skip_num,
                                len(tasks_to_process),
                                output_name,
                                format_skip_details("unchanged"),
                            )
                        )
                        last_info_log_time[0] = time.time()
                        skipped += 1
                        continue

                # Check if file is locked by another process
                if not acquire_lock(output_path, provider.__class__.__name__):
                    output_name = Path(output_path).name
                    skip_num = skipped + completed + failed + 1
                    logger.info(
                        format_task_status(
                            "⏭️ ",
                            skip_num,
                            len(tasks_to_process),
                            output_name,
                            format_skip_details("locked"),
                        )
                    )
                    last_info_log_time[0] = time.time()
                    skipped += 1
                    continue

                # Submit task for processing (lock will be released after task completes)
                # Bind loop variables using default arguments to avoid closure issues
                task_num_counter[0] += 1
                current_task_num = task_num_counter[0]

                def task_with_lock_cleanup(
                    _task_num=current_task_num,
                    _prompt_path=prompt_path,
                    _output_path=output_path,
                    _prompt_hash=prompt_hash,
                ):
                    try:
                        # Log start with full context
                        output_name = Path(_output_path).name
                        prompt_length = Path(_prompt_path).stat().st_size
                        provider_info = provider.get_provider_info()
                        max_tokens = getattr(provider, "max_tokens", "unknown")
                        logger.info(
                            format_task_status(
                                "🚀",
                                _task_num,
                                len(tasks_to_process),
                                output_name,
                                format_start_details(
                                    provider_info["model"], prompt_length, max_tokens
                                ),
                            )
                        )
                        last_info_log_time[0] = time.time()

                        return process_single_task(
                            provider, _prompt_path, _output_path, _prompt_hash
                        )
                    finally:
                        release_lock(_output_path)

                future = executor.submit(task_with_lock_cleanup)
                future_to_task[future] = (prompt_path, output_path, prompt_hash)

            # Process results as they complete
            for future in as_completed(future_to_task):
                output_name, success, error, elapsed, usage, generation_id = (
                    future.result()
                )

                if success:
                    completed += 1

                    # Format log message with optional tokens and tok/s
                    log_msg = format_task_status(
                        "✅",
                        completed + skipped,
                        len(tasks_to_process),
                        output_name,
                        format_success_details(elapsed, usage),
                    )
                    debug_msg = f"Task completed: {output_name} ({format_success_details(elapsed, usage)}"
                    if generation_id:
                        debug_msg += f", gen_id={generation_id}"
                    debug_msg += ")"

                    logger.info(log_msg)
                    logger.debug(debug_msg)
                    last_info_log_time[0] = time.time()
                else:
                    failed += 1
                    logger.error(
                        format_task_status(
                            "❌",
                            completed + failed + skipped,
                            len(tasks_to_process),
                            output_name,
                            f"{elapsed:.1f}s",
                            error,
                        )
                    )
                    logger.debug(
                        f"Task failed: {output_name} - {error} ({elapsed:.1f}s)"
                    )
                    last_info_log_time[0] = time.time()

    # Summary
    logger.info("")
    if skipped > 0:
        logger.info(
            f"📊 Results: {completed} succeeded, {failed} failed, {skipped} skipped, {total_tasks} total"
        )
    else:
        logger.info(
            f"📊 Results: {completed} succeeded, {failed} failed, {total_tasks} total"
        )

    # Log final summary
    logger.debug(
        f"Batch complete: {completed} succeeded, {failed} failed, "
        f"{skipped} skipped, {total_tasks} total"
    )

    return 0 if failed == 0 else 1


def run_single_mode(provider: ModelProvider, verbose: bool) -> int:
    """Run in single-prompt mode, processing prompt from stdin.

    Args:
    ----
        provider: Model provider instance
        verbose: Enable verbose logging

    Returns:
    -------
        Exit code (0 for success)

    """
    # Set up logging
    logger = setup_logging()

    # Ensure unbuffered stdout to avoid truncation
    sys.stdout.reconfigure(line_buffering=True)
    # Read prompt from stdin
    if sys.stdin.isatty():
        logger.error("Error: No input provided. Pipe prompt via stdin.")
        logger.error("Usage: echo 'prompt' | pulseflow-dev model-run --provider openai")
        return 1

    prompt = sys.stdin.read()
    if not prompt.strip():
        logger.error("Error: Empty prompt provided.")
        return 1

    # Get provider and run
    try:
        if verbose:
            logger.info(f"Using provider: {provider.__class__.__name__}")
            if provider.model_name:
                logger.info(f"Using model: {provider.model_name}")

        # Log single mode execution
        logger.debug(
            f"Single mode: provider={provider.__class__.__name__}, model={provider.model_name}"
        )

        # Use async with streaming for LiteLLM providers, sync for CLI providers
        has_async = hasattr(provider, "async_run")
        if has_async:

            async def run_async() -> tuple[str, dict | None, str | None]:
                return await provider.async_run(prompt, stream_to_stdout=True)  # type: ignore[attr-defined]

            result = asyncio.run(run_async())
            # Extract content from tuple (content already streamed to stdout)
            _content, usage, generation_id = result

            # Log usage info if available
            if usage and usage.get("total_tokens"):
                logger.debug(
                    f"Usage: {usage['total_tokens']} tokens "
                    f"(prompt: {usage.get('prompt_tokens')}, completion: {usage.get('completion_tokens')})"
                )
            if generation_id:
                logger.debug(f"Generation ID: {generation_id}")
        else:
            response = provider.run(prompt)
            # Write response to stdout without truncation
            sys.stdout.write(response)
            sys.stdout.flush()

        logger.debug("Single mode: execution completed successfully")
        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        logger.exception(f"Single mode failed: {e}")
        return 1
