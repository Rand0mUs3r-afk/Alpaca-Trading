"""Wiring tests for the backtest WebUI panel and callbacks."""

import unittest

import pandas as pd


class BacktestPanelWiringTests(unittest.TestCase):
    def test_panel_component_builds(self):
        from webui.components.backtest_panel import create_backtest_panel

        panel = create_backtest_panel()
        rendered = str(panel)
        for component_id in (
            "backtest-symbol-input",
            "backtest-run-btn",
            "backtest-equity-graph",
            "backtest-windows-table",
            "backtest-slippage-model",
            "backtest-teach-btn",
            "backtest-teach-status",
        ):
            self.assertIn(component_id, rendered)

    def test_callbacks_register_on_fresh_app(self):
        import dash

        from webui.callbacks.backtest_callbacks import register_backtest_callbacks

        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        register_backtest_callbacks(app)
        # Callback map keys are derived from outputs.
        self.assertTrue(
            any("backtest-status" in key for key in app.callback_map),
            f"backtest callback missing from callback map: {list(app.callback_map)}",
        )
        self.assertTrue(
            any("backtest-teach-status" in key for key in app.callback_map),
            f"teach callback missing from callback map: {list(app.callback_map)}",
        )

    def test_metric_formatting_helpers(self):
        from webui.callbacks.backtest_callbacks import _fmt_pct, _fmt_ratio

        self.assertEqual(_fmt_pct(0.25), "+25.00%")
        self.assertEqual(_fmt_pct(-0.031), "-3.10%")
        self.assertEqual(_fmt_pct(None), "—")
        self.assertEqual(_fmt_ratio(1.234), "1.23")
        self.assertEqual(_fmt_ratio(None), "—")

    def test_equity_figure_and_windows_table_render(self):
        from webui.callbacks.backtest_callbacks import (
            _build_equity_figure,
            _build_windows_table,
        )

        curve = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range("2026-01-05", periods=3),
        )
        figure = _build_equity_figure(curve, "AAPL")
        self.assertEqual(len(figure.data), 1)

        window = {
            "start_date": "2026-01-05",
            "end_date": "2026-02-05",
            "bars": 21,
            "metrics": {
                "cumulative_return": 0.05,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.02,
                "win_rate": 0.6,
            },
        }
        # A single window adds nothing beyond the full-period metrics.
        self.assertIsNone(_build_windows_table([window]))
        self.assertIsNotNone(_build_windows_table([window, window]))


if __name__ == "__main__":
    unittest.main()
