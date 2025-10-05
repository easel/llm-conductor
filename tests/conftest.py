"""Pytest configuration for llm-conductor tests."""

import pytest


@pytest.fixture
def anyio_backend():
    """Configure anyio to only use asyncio backend, not trio."""
    return "asyncio"
