from __future__ import annotations

from typing import Dict

from memory.supabase_client import pooled_cursor


class TradingInterface:
    """Minimal integration point for trading workflows without embedding policy/personality."""

    def get_open_positions(self) -> Dict[str, object]:
        try:
            with pooled_cursor(commit=False) as cur:
                cur.execute(
                    """
                    SELECT id, symbol, side, quantity, entry_price, status, created_at
                    FROM trading_positions
                    WHERE status IN ('open', 'active')
                    ORDER BY created_at DESC
                    LIMIT 50
                    """
                )
                rows = cur.fetchall()
        except Exception:
            return {"positions": []}
        return {
            "positions": [
                {
                    "id": r[0],
                    "symbol": r[1],
                    "side": r[2],
                    "quantity": float(r[3] or 0),
                    "entry_price": float(r[4] or 0),
                    "status": r[5],
                    "created_at": r[6].isoformat() if getattr(r[6], "isoformat", None) else str(r[6]),
                }
                for r in rows
            ]
        }
