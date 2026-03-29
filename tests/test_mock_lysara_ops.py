import unittest
from copy import deepcopy

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


if __name__ == "__main__":
    unittest.main()
