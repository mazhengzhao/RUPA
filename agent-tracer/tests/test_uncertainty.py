"""
Unit tests for uncertainty metrics module.

Tests the TRACER framework uncertainty calculation functions including:
- Normalized entropy calculation
- Edge case handling
- Token-level uncertainties
- Response statistics
"""

import numpy as np
import pytest

from tau2.metrics.uncertainty import (
    TokenUncertainty,
    UncertaintyStats,
    calculate_entropy,
    calculate_normalized_entropy,
    calculate_token_uncertainties,
    get_uncertainty_stats,
)


class TestNormalizedEntropy:
    """Test normalized entropy calculations."""

    def test_basic_calculation(self):
        """Test basic normalized entropy calculation."""
        logprobs = {
            "content": [
                {"token": "I", "logprob": -0.07418411},
                {"token": " can", "logprob": -0.000753561},
                {"token": " help", "logprob": -0.0003308516},
                {"token": " you", "logprob": -0.003733556},
                {"token": " with", "logprob": -0.011826799},
                {"token": " that", "logprob": -0.002088951},
                {"token": ".", "logprob": -0.003147016},
            ]
        }

        ui = calculate_normalized_entropy(logprobs)

        # Manual calculation
        manual_sum = (
            0.07418411
            + 0.000753561
            + 0.0003308516
            + 0.003733556
            + 0.011826799
            + 0.002088951
            + 0.003147016
        )
        manual_ui = manual_sum / 7

        assert abs(ui - manual_ui) < 1e-6, "Normalized entropy calculation mismatch"
        assert ui > 0, "Normalized entropy should be positive"

    def test_none_input(self):
        """Test handling of None input."""
        ui = calculate_normalized_entropy(None)
        assert ui == 0.0, "None input should return 0.0"

    def test_empty_content(self):
        """Test handling of empty content."""
        ui = calculate_normalized_entropy({"content": []})
        assert ui == 0.0, "Empty content should return 0.0"

    def test_single_token(self):
        """Test calculation with a single token."""
        logprobs = {"content": [{"token": "Hello", "logprob": -0.5}]}
        ui = calculate_normalized_entropy(logprobs)
        assert abs(ui - 0.5) < 1e-6, "Single token calculation incorrect"

    def test_high_confidence_response(self):
        """Test very confident response (very low uncertainty)."""
        logprobs = {
            "content": [
                {"token": "Yes", "logprob": -0.0001},
                {"token": ".", "logprob": -0.0001},
            ]
        }
        ui = calculate_normalized_entropy(logprobs)
        assert ui < 0.01, "High confidence should have low uncertainty"

    def test_low_confidence_response(self):
        """Test uncertain response (high uncertainty)."""
        logprobs = {
            "content": [
                {"token": "Maybe", "logprob": -1.5},
                {"token": "...", "logprob": -2.0},
            ]
        }
        ui = calculate_normalized_entropy(logprobs)
        assert ui > 1.0, "Low confidence should have high uncertainty"


class TestTotalEntropy:
    """Test total entropy (unnormalized) calculations."""

    def test_basic_calculation(self):
        """Test basic entropy calculation."""
        logprobs = {
            "content": [
                {"token": "A", "logprob": -0.1},
                {"token": "B", "logprob": -0.2},
                {"token": "C", "logprob": -0.3},
            ]
        }
        entropy = calculate_entropy(logprobs)
        expected = 0.1 + 0.2 + 0.3
        assert abs(entropy - expected) < 1e-6, "Total entropy calculation incorrect"

    def test_none_input(self):
        """Test handling of None input."""
        entropy = calculate_entropy(None)
        assert entropy == 0.0, "None input should return 0.0"


class TestTokenUncertainties:
    """Test token-level uncertainty calculations."""

    def test_basic_calculation(self):
        """Test basic token-level calculation."""
        logprobs = {
            "content": [
                {"token": "Hello", "logprob": -0.1},
                {"token": " world", "logprob": -0.2},
            ]
        }
        uncertainties = calculate_token_uncertainties(logprobs)

        assert len(uncertainties) == 2, "Should have 2 token uncertainties"
        assert isinstance(
            uncertainties[0], TokenUncertainty
        ), "Should return TokenUncertainty objects"

        # Check first token
        assert uncertainties[0].token == "Hello"
        assert abs(uncertainties[0].logprob - (-0.1)) < 1e-6
        assert abs(uncertainties[0].neg_log_likelihood - 0.1) < 1e-6
        assert abs(uncertainties[0].probability - np.exp(-0.1)) < 1e-6

    def test_none_input(self):
        """Test handling of None input."""
        uncertainties = calculate_token_uncertainties(None)
        assert uncertainties == [], "None input should return empty list"

    def test_probability_calculation(self):
        """Test probability calculation from logprob."""
        logprobs = {"content": [{"token": "test", "logprob": -0.0}]}
        uncertainties = calculate_token_uncertainties(logprobs)

        assert len(uncertainties) == 1
        assert abs(uncertainties[0].probability - 1.0) < 1e-6, "P(token) should be 1.0"


class TestUncertaintyStats:
    """Test comprehensive uncertainty statistics."""

    def test_basic_stats(self):
        """Test basic statistics calculation."""
        logprobs = {
            "content": [
                {"token": "Hi", "logprob": -0.023732105},
                {"token": " there", "logprob": -0.62046385},
                {"token": "!", "logprob": -0.32751456},
            ]
        }
        stats = get_uncertainty_stats(logprobs)

        assert isinstance(stats, UncertaintyStats), "Should return UncertaintyStats"
        assert stats.token_count == 3, "Should have 3 tokens"
        assert stats.normalized_entropy > 0, "Normalized entropy should be positive"
        assert stats.total_entropy > 0, "Total entropy should be positive"
        assert (
            stats.min_uncertainty < stats.max_uncertainty
        ), "Min should be less than max"

        # Verify normalized entropy calculation
        expected_normalized = (0.023732105 + 0.62046385 + 0.32751456) / 3
        assert (
            abs(stats.normalized_entropy - expected_normalized) < 1e-6
        ), "Normalized entropy mismatch"

    def test_none_input(self):
        """Test handling of None input."""
        stats = get_uncertainty_stats(None)
        assert stats.token_count == 0, "None input should have 0 tokens"
        assert stats.normalized_entropy == 0.0, "Should have 0 entropy"

    def test_mean_probability(self):
        """Test mean probability calculation."""
        logprobs = {
            "content": [
                {"token": "A", "logprob": -0.1},
                {"token": "B", "logprob": -0.1},
            ]
        }
        stats = get_uncertainty_stats(logprobs)

        expected_prob = np.exp(-0.1)
        assert (
            abs(stats.mean_probability - expected_prob) < 1e-6
        ), "Mean probability incorrect"


class TestRealWorldData:
    """Test with real-world data structures from Gemini/VertexAI."""

    def test_gemini_format(self):
        """Test with actual Gemini logprobs format."""
        # This is the exact structure from Gemini responses
        real_world_logprobs = {
            "content": [
                {
                    "token": "Hi",
                    "bytes": None,
                    "logprob": -0.023732105,
                    "top_logprobs": [],
                },
                {
                    "token": " there",
                    "bytes": None,
                    "logprob": -0.62046385,
                    "top_logprobs": [],
                },
                {
                    "token": "!",
                    "bytes": None,
                    "logprob": -0.32751456,
                    "top_logprobs": [],
                },
            ]
        }

        ui = calculate_normalized_entropy(real_world_logprobs)
        stats = get_uncertainty_stats(real_world_logprobs)

        assert 0.0 < ui < 2.0, "U_i should be in reasonable range"
        assert stats.token_count == 3, "Should have 3 tokens"
        assert stats.mean_probability > 0, "Mean probability should be positive"
        assert stats.mean_probability <= 1.0, "Mean probability should be <= 1.0"

    def test_long_response(self):
        """Test with a longer response (similar to real conversations)."""
        # Simulate a 20-token response
        logprobs = {
            "content": [
                {"token": f"token{i}", "logprob": -0.05 * (i % 5)} for i in range(20)
            ]
        }

        ui = calculate_normalized_entropy(logprobs)
        stats = get_uncertainty_stats(logprobs)

        assert stats.token_count == 20, "Should have 20 tokens"
        assert ui >= 0, "U_i should be non-negative"
        assert stats.std_uncertainty >= 0, "Std should be non-negative"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_missing_logprob_field(self):
        """Test handling of missing logprob field in token data."""
        logprobs = {
            "content": [
                {"token": "A", "logprob": -0.1},
                {"token": "B"},  # Missing logprob
                {"token": "C", "logprob": -0.2},
            ]
        }
        ui = calculate_normalized_entropy(logprobs)
        # Should only count tokens with logprob
        expected = (0.1 + 0.2) / 2
        assert abs(ui - expected) < 1e-6, "Should skip tokens without logprob"

    def test_zero_logprob(self):
        """Test token with zero logprob (probability = 1.0)."""
        logprobs = {"content": [{"token": "certain", "logprob": 0.0}]}
        ui = calculate_normalized_entropy(logprobs)
        assert ui == 0.0, "Zero logprob should give zero uncertainty"

    def test_very_negative_logprob(self):
        """Test token with very negative logprob (very uncertain)."""
        logprobs = {"content": [{"token": "uncertain", "logprob": -10.0}]}
        ui = calculate_normalized_entropy(logprobs)
        assert ui == 10.0, "Very negative logprob should give high uncertainty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

