"""Base class for LiteLLM-based model providers with streaming support."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from abc import abstractmethod

from llm_conductor.base import ModelProvider

# Configuration constants
DEFAULT_MAX_TOKENS = 8192  # Optimized for modern models (Claude 3.5, GPT-4o)

# Suppress LiteLLM debug output by default (unless LITELLM_DEBUG is set)
if not os.getenv("LITELLM_DEBUG"):
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False


class LiteLLMProvider(ModelProvider):
    """Abstract base class for LiteLLM-based API providers.

    Provides shared async_run() implementation with streaming and auto-continuation
    for OpenAI, LMStudio, and OpenRouter providers.
    """

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name)
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        self.num_retries = int(os.getenv("LITELLM_NUM_RETRIES", "3"))
        self.timeout = int(os.getenv("LITELLM_TIMEOUT", "120"))
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def _get_api_key(self) -> str:
        """Get the API key for this provider."""

    @abstractmethod
    def _get_base_url(self) -> str | None:
        """Get the base URL for this provider (None uses default)."""

    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model name for this provider."""

    def _get_model_prefix(self) -> str:
        """Get the model name prefix (e.g., 'openrouter/' for OpenRouter)."""
        return ""

    def _prepare_model_name(self, model_name: str) -> str:
        """Prepare the model name with provider-specific prefix."""
        prefix = self._get_model_prefix()
        if prefix and not model_name.startswith(prefix):
            return f"{prefix}{model_name}"
        return model_name

    async def async_run(
        self, prompt: str, stream_to_stdout: bool = False
    ) -> tuple[str, dict | None, str | None]:
        """Run LiteLLM API with streaming and auto-continuation for truncated responses.

        Always uses streaming mode for better long-response handling and faster continuation.
        Accumulates full response while optionally writing to stdout.

        Args:
        ----
            prompt: The prompt text to send to the model
            stream_to_stdout: If True, write chunks to stdout while accumulating

        Returns:
        -------
            Tuple of (content, usage_dict, generation_id):
                - content: The complete model response as a string
                - usage_dict: Token usage info (prompt_tokens, completion_tokens, total_tokens) or None
                - generation_id: Provider-specific generation/request ID or None

        """
        try:
            import litellm

            # Enable debug mode if LITELLM_DEBUG is explicitly set
            if os.getenv("LITELLM_DEBUG"):
                litellm._turn_on_debug()  # type: ignore[attr-defined]  # noqa: SLF001
        except ImportError as e:
            msg = "LiteLLM not installed. Install with: uv add litellm"
            raise RuntimeError(msg) from e

        # Application-level retry for empty responses
        max_app_retries = int(os.getenv("LITELLM_APP_RETRIES", "3"))
        logger = logging.getLogger(__name__)

        for attempt in range(max_app_retries):
            try:
                # Set model name with prefix
                if not self.model_name:
                    self.model_name = self._get_default_model()
                model = self._prepare_model_name(self.model_name)

                # Initial message
                messages = [{"role": "user", "content": prompt}]
                full_content = ""
                all_chunks: list = []  # Collect chunks for metadata extraction

                # Log request start
                prompt_length = len(prompt)
                self.logger.debug(
                    f"🚀 Starting {model} request (prompt: {prompt_length} chars, max_tokens: {self.max_tokens})"
                )

                # Auto-continuation loop with streaming
                request_start_time = asyncio.get_event_loop().time()
                continuation_count = 0
                while True:
                    stream = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        api_key=self._get_api_key(),
                        api_base=self._get_base_url(),
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=True,
                        num_retries=self.num_retries,
                        timeout=self.timeout,
                    )

                    content = ""
                    finish_reason = None
                    chunk_errors = 0
                    chunk_count = 0
                    last_progress_log = asyncio.get_event_loop().time()

                    # Accumulate chunks (and optionally write to stdout)
                    async for chunk in stream:
                        chunk_count += 1
                        all_chunks.append(chunk)  # Collect for metadata
                        try:
                            if not chunk:
                                continue

                            # Access choices - handle both object attributes and dict keys
                            choices = None

                            # Try attribute access first
                            with contextlib.suppress(AttributeError, TypeError):
                                choices = getattr(chunk, "choices", None)

                            # If attribute access didn't work or returned None, try dict access
                            if choices is None:
                                with contextlib.suppress(AttributeError, TypeError):
                                    choices = (
                                        chunk.get("choices")
                                        if hasattr(chunk, "get")
                                        else None
                                    )

                            # If still no choices, try direct indexing (for unusual chunk formats)
                            if choices is None:
                                with contextlib.suppress(
                                    KeyError, TypeError, IndexError
                                ):
                                    choices = chunk["choices"]

                            if not choices:
                                continue

                            choice = choices[0]

                            # Capture finish_reason if present (try all access methods)
                            finish_reason_val = None
                            with contextlib.suppress(AttributeError, TypeError):
                                finish_reason_val = getattr(
                                    choice, "finish_reason", None
                                )
                            if finish_reason_val is None:
                                with contextlib.suppress(AttributeError, TypeError):
                                    finish_reason_val = (
                                        choice.get("finish_reason")
                                        if hasattr(choice, "get")
                                        else None
                                    )
                            if finish_reason_val is None:
                                with contextlib.suppress(
                                    KeyError, TypeError, IndexError
                                ):
                                    finish_reason_val = choice["finish_reason"]
                            if finish_reason_val:
                                finish_reason = finish_reason_val

                            # Extract content from delta (try all access methods)
                            delta = None
                            with contextlib.suppress(AttributeError, TypeError):
                                delta = getattr(choice, "delta", None)
                            if delta is None:
                                with contextlib.suppress(AttributeError, TypeError):
                                    delta = (
                                        choice.get("delta")
                                        if hasattr(choice, "get")
                                        else None
                                    )
                            if delta is None:
                                with contextlib.suppress(
                                    KeyError, TypeError, IndexError
                                ):
                                    delta = choice["delta"]

                            if delta:
                                delta_content = None
                                with contextlib.suppress(AttributeError, TypeError):
                                    delta_content = getattr(delta, "content", None)
                                if delta_content is None:
                                    with contextlib.suppress(AttributeError, TypeError):
                                        delta_content = (
                                            delta.get("content")
                                            if hasattr(delta, "get")
                                            else None
                                        )
                                if delta_content is None:
                                    with contextlib.suppress(
                                        KeyError, TypeError, IndexError
                                    ):
                                        delta_content = delta["content"]

                                if delta_content:
                                    content += delta_content
                                    if stream_to_stdout:
                                        sys.stdout.write(delta_content)
                                        sys.stdout.flush()

                                    # Log progress every 10 seconds
                                    current_time = asyncio.get_event_loop().time()
                                    if current_time - last_progress_log >= 10.0:
                                        elapsed = current_time - request_start_time
                                        tokens_so_far = len(full_content) + len(content)
                                        self.logger.info(
                                            f"⏳ Streaming... {chunk_count} chunks, ~{tokens_so_far} chars, {elapsed:.1f}s elapsed"
                                        )
                                        last_progress_log = current_time
                        except Exception as e:
                            # Skip malformed chunks but track errors
                            chunk_errors += 1
                            if chunk_errors <= 3:  # Log first few errors
                                logging.getLogger(__name__).warning(
                                    f"Chunk parsing error ({type(e).__name__}: {e})"
                                )
                            continue

                    # Warn if we had chunk errors but got some content
                    if chunk_errors > 0 and content:
                        logging.getLogger(__name__).warning(
                            f"Encountered {chunk_errors} chunk parsing errors but recovered partial content"
                        )

                    # Only fail if we got absolutely nothing
                    if not content and not full_content:
                        if chunk_errors > 0:
                            msg = f"{self.__class__.__name__} returned empty response after {chunk_errors} chunk parsing errors"
                        else:
                            msg = f"{self.__class__.__name__} returned empty response"
                        raise RuntimeError(msg)

                    full_content += content

                    # Check if we need to continue
                    if finish_reason == "length":
                        continuation_count += 1
                        self.logger.info(
                            f"🔄 Response truncated (continuation {continuation_count}), requesting more..."
                        )
                        # Append assistant's response with accumulated content and request continuation
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": "continue"})
                        continue  # Restart loop for continuation
                    # Response is complete
                    break

                # Extract usage and generation ID from collected chunks (OUTSIDE while loop)
                usage_dict = None
                generation_id = None

                try:
                    # Rebuild full response with metadata using stream_chunk_builder
                    full_response = litellm.stream_chunk_builder(
                        all_chunks, messages=messages
                    )

                    # Extract usage information
                    if hasattr(full_response, "usage"):
                        usage_obj = full_response.usage
                        usage_dict = {
                            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                            "completion_tokens": getattr(
                                usage_obj, "completion_tokens", None
                            ),
                            "total_tokens": getattr(usage_obj, "total_tokens", None),
                        }
                    elif isinstance(full_response, dict) and "usage" in full_response:
                        usage_dict = full_response["usage"]

                    # Calculate actual cost using LiteLLM
                    if usage_dict:
                        try:
                            cost = litellm.completion_cost(
                                completion_response=full_response
                            )
                            if cost is not None:
                                usage_dict["cost_usd"] = round(cost, 6)
                                usage_dict["cost_calculation_method"] = "litellm_actual"
                        except Exception:
                            # If cost calculation fails, it's not critical
                            pass

                    # Extract generation ID
                    if hasattr(full_response, "id"):
                        generation_id = full_response.id
                    elif isinstance(full_response, dict) and "id" in full_response:
                        generation_id = full_response["id"]
                except Exception as e:
                    # Log but don't fail if metadata extraction fails
                    logging.getLogger(__name__).debug(
                        f"Failed to extract metadata: {e}"
                    )

                # Log completion
                total_elapsed = asyncio.get_event_loop().time() - request_start_time
                content_length = len(full_content)
                self.logger.debug(
                    f"✅ Completed {model} request ({content_length} chars, {total_elapsed:.1f}s)"
                )
                if usage_dict and usage_dict.get("total_tokens"):
                    tok_per_sec = (
                        usage_dict["total_tokens"] / total_elapsed
                        if total_elapsed > 0
                        else 0
                    )
                    self.logger.debug(
                        f"   Tokens: {usage_dict['prompt_tokens']}→{usage_dict['completion_tokens']} "
                        f"({usage_dict['total_tokens']} total, {tok_per_sec:.0f} tok/s)"
                    )

                return (full_content, usage_dict, generation_id)

            except RuntimeError as e:
                # Retry on empty response errors
                if (
                    "returned empty response" in str(e)
                    and attempt < max_app_retries - 1
                ):
                    backoff_seconds = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"{self.__class__.__name__}: Empty response on attempt {attempt + 1}/{max_app_retries}, "
                        f"retrying in {backoff_seconds}s..."
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue
                # Re-raise if not empty response error or last attempt
                msg = f"{self.__class__.__name__} API call failed: {e}"
                raise RuntimeError(msg) from e
            except Exception as e:
                msg = f"{self.__class__.__name__} API call failed: {e}"
                raise RuntimeError(msg) from e

        # Should never reach here (all paths return or raise in loop)
        msg = f"{self.__class__.__name__}: All retry attempts exhausted"
        raise RuntimeError(msg)
