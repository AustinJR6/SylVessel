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


if __name__ == "__main__":
    unittest.main()
