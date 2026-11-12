"""Tests for deterministic market-regime detection.

Three cheap observable dimensions — realized-volatility percentile (with an
absolute annualized floor), price-vs-SMA trend, and dollar-volume liquidity
— combine into a regime label and a risk multiplier that only ever shrinks
position sizes. A regime filter, not a signal generator: hostile regimes
reduce size, they never flip decisions. Missing data reads as unknown with
multiplier 1.0.
"""

import unittest

import numpy as np
import pandas as pd

from tradingagents.regime import (
    RegimeConfig,
    classify_regime,
    regime_report_block,
    regime_risk_multiplier,
)


def _frame(closes, volumes=None, start="2024-01-01"):
    closes = list(closes)
    dates = pd.bdate_range(start=start, periods=len(closes))
    volumes = volumes if volumes is not None else [1_000_000.0] * len(closes)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": closes,
            "high": [c * 1.005 for c in closes],
            "low": [c * 0.995 for c in closes],
            "close": closes,
            "volume": volumes,
        }
    )


def _calm_uptrend(n=300, seed=5):
    rng = np.random.default_rng(seed)
    return list(100 * np.cumprod(1 + 0.0008 + rng.normal(0, 0.004, n)))


def _calm_then_crash(n_calm=260, n_crash=40, seed=9):
    rng = np.random.default_rng(seed)
    calm = 0.0005 + rng.normal(0, 0.004, n_calm)
    crash = -0.02 + rng.normal(0, 0.035, n_crash)
    return list(100 * np.cumprod(1 + np.concatenate([calm, crash])))


class ClassificationTests(unittest.TestCase):
    def test_calm_uptrend_is_favorable_full_size(self):
        assessment = classify_regime(_frame(_calm_uptrend()), symbol="SPY")
        self.assertEqual(assessment.trend_state, "up")
        self.assertIn(assessment.volatility_state, ("calm", "normal"))
        self.assertEqual(assessment.label, "favorable")
        self.assertEqual(assessment.risk_multiplier, 1.0)

    def test_volatile_crash_is_hostile_and_shrinks_size(self):
        assessment = classify_regime(_frame(_calm_then_crash()), symbol="SPY")
        self.assertEqual(assessment.volatility_state, "turbulent")
        self.assertEqual(assessment.trend_state, "down")
        self.assertEqual(assessment.label, "hostile")
        # 0.5 (turbulent) * 0.75 (downtrend) = 0.375
        self.assertAlmostEqual(assessment.risk_multiplier, 0.375)

    def test_persistently_wild_asset_hits_absolute_vol_floor(self):
        # Constant 4% daily noise: its own percentile history is unremarkable,
        # but ~63% annualized vol must still read as turbulent.
        rng = np.random.default_rng(2)
        closes = list(100 * np.cumprod(1 + rng.normal(0, 0.04, 300)))
        assessment = classify_regime(_frame(closes), symbol="MEME")
        self.assertEqual(assessment.volatility_state, "turbulent")

    def test_thinning_liquidity_is_flagged_and_penalized(self):
        closes = _calm_uptrend()
        volumes = [1_000_000.0] * 240 + [200_000.0] * 60
        assessment = classify_regime(_frame(closes, volumes), symbol="SMALL")
        self.assertEqual(assessment.liquidity_state, "thinning")
        self.assertLess(assessment.risk_multiplier, 1.0)

    def test_insufficient_history_reads_unknown_full_size(self):
        assessment = classify_regime(_frame(_calm_uptrend(10)), symbol="IPO")
        self.assertEqual(assessment.volatility_state, "unknown")
        self.assertEqual(assessment.risk_multiplier, 1.0)

    def test_multiplier_never_below_floor(self):
        closes = _calm_then_crash()
        volumes = [1_000_000.0] * 240 + [100_000.0] * 60
        config = RegimeConfig(min_risk_multiplier=0.3)
        assessment = classify_regime(_frame(closes, volumes), config=config)
        self.assertGreaterEqual(assessment.risk_multiplier, 0.3)

    def test_markdown_block_names_all_three_dimensions(self):
        assessment = classify_regime(_frame(_calm_uptrend()), symbol="SPY")
        text = assessment.to_markdown()
        for needle in ("Regime", "Volatility", "Trend", "Liquidity"):
            self.assertIn(needle, text)


class InjectionHelperTests(unittest.TestCase):
    def test_report_block_uses_injected_loader(self):
        block = regime_report_block(
            "SPY", price_loader=lambda symbol, start, end: _frame(_calm_uptrend())
        )
        self.assertIn("Market Regime", block)
        self.assertIn("favorable", block)

    def test_report_block_swallows_loader_failure(self):
        def broken(symbol, start, end):
            raise ConnectionError("data source down")

        self.assertEqual(regime_report_block("SPY", price_loader=broken), "")

    def test_risk_multiplier_helper(self):
        multiplier = regime_risk_multiplier(
            "SPY", price_loader=lambda symbol, start, end: _frame(_calm_then_crash())
        )
        self.assertAlmostEqual(multiplier, 0.375)

    def test_risk_multiplier_helper_failure_is_neutral(self):
        def broken(symbol, start, end):
            raise ConnectionError("data source down")

        self.assertEqual(regime_risk_multiplier("SPY", price_loader=broken), 1.0)


class ConfigTests(unittest.TestCase):
    def test_default_config_exposes_regime_keys(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        self.assertIn("regime_detection_enabled", DEFAULT_CONFIG)
        self.assertIn("regime_turbulent_size_factor", DEFAULT_CONFIG)

    def test_from_config_reads_project_keys(self):
        config = RegimeConfig.from_config(
            {"regime_turbulent_size_factor": 0.4, "regime_trend_window": 30}
        )
        self.assertEqual(config.turbulent_size_factor, 0.4)
        self.assertEqual(config.trend_window, 30)


if __name__ == "__main__":
    unittest.main()
