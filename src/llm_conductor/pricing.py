"""Model pricing data and cost calculation utilities.

Pricing is in USD per million tokens. Data sourced from official provider pricing pages
and updated regularly. Last updated: January 2025.
"""

from __future__ import annotations

# Model pricing in USD per million tokens (input/output)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude models (via Anthropic API)
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-0613": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    # OpenAI o1 models (reasoning models with higher costs)
    "o1": {"input": 15.00, "output": 60.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    # Estimated pricing for GPT-5 (not released yet, placeholder)
    "gpt-5": {"input": 5.00, "output": 15.00},
    # Default fallback (mid-range pricing)
    "default": {"input": 5.00, "output": 15.00},
}


def get_model_pricing(model_name: str) -> dict[str, float]:
    """Get pricing for a model by name.

    Args:
    ----
        model_name: The model identifier

    Returns:
    -------
        Dictionary with 'input' and 'output' prices per million tokens

    """
    # Try exact match first
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Try partial match (e.g., "claude-3-5-sonnet-latest" -> "claude-3-5-sonnet")
    for model_key, pricing in MODEL_PRICING.items():
        if model_key in model_name or model_name in model_key:
            return pricing

    # Fallback to default
    return MODEL_PRICING["default"]


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> float:
    """Calculate estimated cost in USD based on token counts and model.

    Args:
    ----
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        model: Model identifier

    Returns:
    -------
        Estimated cost in USD (rounded to 6 decimal places)

    """
    pricing = get_model_pricing(model)

    # Calculate costs (pricing is per million tokens)
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["output"]
    total_cost = prompt_cost + completion_cost

    # Round to 6 decimal places (sufficient precision for cost tracking)
    return round(total_cost, 6)
