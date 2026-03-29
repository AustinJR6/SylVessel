import unittest
from copy import deepcopy
from unittest.mock import patch

from scripts import mock_lysara_ops


class MockLysaraOpsTests(unittest.TestCase):
    def test_build_exposure_payload_reflects_open_positions(self):
        state = deepcopy(mock_lysara_ops._default_state(starting_balance=1000.0))
        state["positions"]["items"] = [
            {
                "symbol": "BTC-USD",
                "market": "crypto",
                "quantity": 0.0005,
                "avg_entry_price": 68000.0,
                "created_at": "2026-03-29T00:00:00+00:00",
                "updated_at": "2026-03-29T00:00:00+00:00",
            }
        ]
        mock_lysara_ops._update_portfolio_snapshot(state)

        payload = mock_lysara_ops._build_exposure_payload(state, "crypto")

        self.assertEqual(payload["market"], "crypto")
        self.assertTrue(payload["mock_mode"])
        self.assertGreater(payload["gross_exposure_pct"], 0.0)
        self.assertEqual(len(payload["positions"]), 1)
        self.assertEqual(payload["positions"][0]["symbol"], "BTC-USD")
        self.assertGreater(payload["positions"][0]["effective_weight_pct"], 0.0)

    def test_upsert_strategy_updates_enabled_state_and_symbols(self):
        original_state = deepcopy(mock_lysara_ops.STATE)
        try:
            mock_lysara_ops.STATE.clear()
            mock_lysara_ops.STATE.update(deepcopy(mock_lysara_ops._default_state(starting_balance=1000.0)))

            result = mock_lysara_ops._upsert_strategy_locked(
                {
                    "market": "crypto",
                    "strategy_name": "MomentumStrategy",
                    "enabled": False,
                    "params": {"lookback": 12, "trade_symbols": ["BTC-USD", "DOGE-USD"]},
                    "symbols": ["BTC-USD", "DOGE-USD"],
                }
            )

            self.assertEqual(result["status"], "updated")
            self.assertFalse(result["strategy"]["enabled"])
            self.assertEqual(result["strategy"]["symbols"], ["BTC-USD", "DOGE-USD"])
            self.assertEqual(
                mock_lysara_ops.STATE["status"]["strategy_controls"]["crypto"]["MomentumStrategy"],
                False,
            )
        finally:
            mock_lysara_ops.STATE.clear()
            mock_lysara_ops.STATE.update(original_state)

    def test_watchlist_update_and_candidate_queue_persist_in_state(self):
        original_state = deepcopy(mock_lysara_ops.STATE)
        try:
            mock_lysara_ops.STATE.clear()
            mock_lysara_ops.STATE.update(deepcopy(mock_lysara_ops._default_state(starting_balance=1000.0)))

            watchlist_payload = mock_lysara_ops._update_watchlists_locked(
                {"crypto": ["BTC-USD", "PEPE-USD"], "stocks": ["AAPL"]}
            )
            queued = mock_lysara_ops._queue_strategy_candidate_locked(
                {
                    "symbol": "PEPE-USD",
                    "market": "crypto",
                    "strategy_key": "MomentumStrategy",
                    "summary": "High-volatility meme candidate",
                }
            )

            self.assertEqual(watchlist_payload["watchlists"]["crypto"], ["BTC-USD", "PEPE-USD"])
            self.assertEqual(queued["status"], "queued")
            self.assertEqual(queued["item"]["symbol"], "PEPE-USD")
            self.assertEqual(mock_lysara_ops.STATE["strategy_candidates"][0]["strategy_key"], "MomentumStrategy")
        finally:
            mock_lysara_ops.STATE.clear()
            mock_lysara_ops.STATE.update(original_state)

    def test_refresh_quotes_uses_yahoo_chart_payloads(self):
        def fake_fetch(url: str, timeout: float = 8.0):
            if "AAPL" in url:
                return {
                    "chart": {
                        "result": [
                            {
                                "meta": {
                                    "regularMarketPrice": 248.8,
                                    "chartPreviousClose": 252.89,
                                }
                            }
                        ]
                    }
                }
            if "BTC-USD" in url:
                return {
                    "chart": {
                        "result": [
                            {
                                "meta": {
                                    "regularMarketPrice": 66565.495,
                                    "chartPreviousClose": 67044.0,
                                }
                            }
                        ]
                    }
                }
            return None

        with patch("scripts.mock_lysara_ops._fetch_json", side_effect=fake_fetch):
            stocks = mock_lysara_ops._refresh_stock_quotes(["AAPL"])
            crypto = mock_lysara_ops._refresh_crypto_quotes(["BTC-USD"])

        self.assertEqual(stocks["AAPL"]["source"], "yahoo_chart")
        self.assertAlmostEqual(stocks["AAPL"]["price"], 248.8)
        self.assertEqual(crypto["BTC-USD"]["source"], "yahoo_chart")
        self.assertAlmostEqual(crypto["BTC-USD"]["price"], 66565.495)

    def test_load_state_reseeds_default_strategies_when_registry_empty(self):
        state = deepcopy(mock_lysara_ops._default_state(starting_balance=1000.0))
        state["status"]["strategy_registry"] = {"stocks": [], "crypto": []}
        changed = mock_lysara_ops._ensure_strategy_registry(state)

        self.assertTrue(changed)
        self.assertGreater(len(state["status"]["strategy_registry"]["stocks"]), 0)
        self.assertGreater(len(state["status"]["strategy_registry"]["crypto"]), 0)


if __name__ == "__main__":
    unittest.main()
