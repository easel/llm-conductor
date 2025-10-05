"""Unit tests for pricing module."""

from __future__ import annotations

import pytest

from llm_conductor.pricing import MODEL_PRICING, calculate_cost, get_model_pricing


class TestGetModelPricing:
    """Test get_model_pricing function."""

    def test_exact_match(self):
        """Test exact model name match."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing == {"input": 2.50, "output": 10.00}

    def test_exact_match_claude(self):
        """Test exact match for Claude model."""
        pricing = get_model_pricing("claude-3-5-sonnet")
        assert pricing == {"input": 3.00, "output": 15.00}

    def test_exact_match_with_version(self):
        """Test exact match with version suffix."""
        pricing = get_model_pricing("gpt-4o-2024-11-20")
        assert pricing == {"input": 2.50, "output": 10.00}

    def test_partial_match(self):
        """Test partial model name matching."""
        # Model name contains known model key
        pricing = get_model_pricing("gpt-4o-latest")
        assert pricing == {"input": 2.50, "output": 10.00}

    def test_partial_match_claude(self):
        """Test partial matching for Claude models."""
        pricing = get_model_pricing("claude-3-haiku-latest")
        assert pricing == {"input": 0.25, "output": 1.25}

    def test_unknown_model_returns_default(self):
        """Test unknown model returns default pricing."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing == MODEL_PRICING["default"]
        assert pricing == {"input": 5.00, "output": 15.00}

    def test_o1_models(self):
        """Test o1 reasoning models have correct pricing."""
        pricing_o1 = get_model_pricing("o1")
        assert pricing_o1 == {"input": 15.00, "output": 60.00}

        pricing_o1_mini = get_model_pricing("o1-mini")
        assert pricing_o1_mini == {"input": 3.00, "output": 12.00}

    def test_gpt_4_turbo(self):
        """Test GPT-4 Turbo pricing."""
        pricing = get_model_pricing("gpt-4-turbo")
        assert pricing == {"input": 10.00, "output": 30.00}

    def test_gpt_4o_mini(self):
        """Test GPT-4o mini (cheap model) pricing."""
        pricing = get_model_pricing("gpt-4o-mini")
        assert pricing == {"input": 0.15, "output": 0.60}

    def test_claude_opus(self):
        """Test Claude Opus (expensive model) pricing."""
        pricing = get_model_pricing("claude-3-opus")
        assert pricing == {"input": 15.00, "output": 75.00}


class TestCalculateCost:
    """Test calculate_cost function."""

    def test_basic_cost_calculation(self):
        """Test basic cost calculation."""
        # Using gpt-4o-mini: input $0.15, output $0.60 per million
        # 1000 input tokens = $0.00015
        # 500 output tokens = $0.0003
        # Total = $0.00045
        cost = calculate_cost(1000, 500, "gpt-4o-mini")
        assert cost == pytest.approx(0.00045, abs=1e-6)

    def test_cost_calculation_gpt4(self):
        """Test cost calculation for GPT-4."""
        # GPT-4: input $30, output $60 per million
        # 10000 input = $0.30
        # 5000 output = $0.30
        # Total = $0.60
        cost = calculate_cost(10000, 5000, "gpt-4")
        assert cost == pytest.approx(0.60, abs=1e-6)

    def test_cost_calculation_claude_sonnet(self):
        """Test cost calculation for Claude Sonnet."""
        # Claude 3.5 Sonnet: input $3, output $15 per million
        # 100000 input = $0.30
        # 50000 output = $0.75
        # Total = $1.05
        cost = calculate_cost(100000, 50000, "claude-3-5-sonnet")
        assert cost == pytest.approx(1.05, abs=1e-6)

    def test_cost_calculation_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = calculate_cost(0, 0, "gpt-4o")
        assert cost == 0.0

    def test_cost_calculation_only_input_tokens(self):
        """Test cost calculation with only input tokens."""
        # gpt-4o: input $2.50 per million
        # 1000 tokens = $0.0025
        cost = calculate_cost(1000, 0, "gpt-4o")
        assert cost == pytest.approx(0.0025, abs=1e-6)

    def test_cost_calculation_only_output_tokens(self):
        """Test cost calculation with only output tokens."""
        # gpt-4o: output $10 per million
        # 1000 tokens = $0.01
        cost = calculate_cost(0, 1000, "gpt-4o")
        assert cost == pytest.approx(0.01, abs=1e-6)

    def test_cost_calculation_large_numbers(self):
        """Test cost calculation with large token counts."""
        # gpt-4o-mini: input $0.15, output $0.60
        # 10M input = $1.50
        # 5M output = $3.00
        # Total = $4.50
        cost = calculate_cost(10_000_000, 5_000_000, "gpt-4o-mini")
        assert cost == pytest.approx(4.50, abs=1e-6)

    def test_cost_calculation_unknown_model_uses_default(self):
        """Test that unknown model uses default pricing."""
        # Default: input $5, output $15 per million
        # 1000 input = $0.005
        # 1000 output = $0.015
        # Total = $0.02
        cost = calculate_cost(1000, 1000, "unknown-model")
        assert cost == pytest.approx(0.02, abs=1e-6)

    def test_cost_rounding(self):
        """Test that costs are rounded to 6 decimal places."""
        # Create a scenario that would have more than 6 decimals
        # 1 token at gpt-4o-mini rates
        cost = calculate_cost(1, 1, "gpt-4o-mini")
        # Should be rounded to 6 decimals
        assert len(str(cost).split(".")[-1]) <= 6

    def test_cost_calculation_o1_models(self):
        """Test cost calculation for o1 reasoning models."""
        # o1: input $15, output $60 per million
        # 10000 input = $0.15
        # 10000 output = $0.60
        # Total = $0.75
        cost = calculate_cost(10000, 10000, "o1")
        assert cost == pytest.approx(0.75, abs=1e-6)

    def test_cost_calculation_haiku(self):
        """Test cost calculation for cheapest model (Haiku)."""
        # Claude Haiku: input $0.25, output $1.25 per million
        # 100000 input = $0.025
        # 50000 output = $0.0625
        # Total = $0.0875
        cost = calculate_cost(100000, 50000, "claude-3-haiku")
        assert cost == pytest.approx(0.0875, abs=1e-6)


class TestModelPricingData:
    """Test MODEL_PRICING data structure integrity."""

    def test_all_models_have_input_and_output_prices(self):
        """Test that all models have both input and output prices."""
        for model_name, pricing in MODEL_PRICING.items():
            assert "input" in pricing, f"{model_name} missing 'input' price"
            assert "output" in pricing, f"{model_name} missing 'output' price"

    def test_all_prices_are_positive(self):
        """Test that all prices are positive numbers."""
        for model_name, pricing in MODEL_PRICING.items():
            assert pricing["input"] > 0, f"{model_name} has non-positive input price"
            assert pricing["output"] > 0, f"{model_name} has non-positive output price"

    def test_output_prices_higher_than_input(self):
        """Test that output prices are typically higher than input prices."""
        # This is a general trend, not a strict rule, but let's verify most models
        higher_output_count = 0
        for _model_name, pricing in MODEL_PRICING.items():
            if pricing["output"] >= pricing["input"]:
                higher_output_count += 1

        # At least 90% of models should follow this pattern
        assert higher_output_count / len(MODEL_PRICING) >= 0.9

    def test_default_pricing_exists(self):
        """Test that default pricing exists."""
        assert "default" in MODEL_PRICING
        assert MODEL_PRICING["default"]["input"] > 0
        assert MODEL_PRICING["default"]["output"] > 0

    def test_pricing_data_types(self):
        """Test that all prices are floats."""
        for model_name, pricing in MODEL_PRICING.items():
            assert isinstance(
                pricing["input"], int | float
            ), f"{model_name} input price is not numeric"
            assert isinstance(
                pricing["output"], int | float
            ), f"{model_name} output price is not numeric"

    def test_known_expensive_model(self):
        """Test that known expensive model (Opus) is priced higher than average."""
        opus_pricing = MODEL_PRICING["claude-3-opus"]
        default_pricing = MODEL_PRICING["default"]

        assert opus_pricing["input"] > default_pricing["input"]
        assert opus_pricing["output"] > default_pricing["output"]

    def test_known_cheap_model(self):
        """Test that known cheap model (Haiku) is priced lower than average."""
        haiku_pricing = MODEL_PRICING["claude-3-haiku"]
        default_pricing = MODEL_PRICING["default"]

        assert haiku_pricing["input"] < default_pricing["input"]
        assert haiku_pricing["output"] < default_pricing["output"]
