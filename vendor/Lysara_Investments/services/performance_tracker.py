import sqlite3
from pathlib import Path
from typing import Dict, List

class PerformanceTracker:
    """Utility class to compute P/L and equity stats from the trade database."""

    def __init__(self, db_path: str = "trades.db"):
        self.db_path = Path(db_path)

    def _connect(self):
        if not self.db_path.is_file():
            return None
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def pnl_per_symbol(self) -> Dict[str, float]:
        conn = self._connect()
        if not conn:
            return {}
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, SUM(profit_loss) FROM trades WHERE profit_loss IS NOT NULL GROUP BY symbol"
        )
        rows = cur.fetchall()
        conn.close()
        return {r[0]: float(r[1] or 0) for r in rows}

    def equity_curve(self, limit: int = 500) -> List[Dict]:
        conn = self._connect()
        if not conn:
            return []
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp, total_equity FROM equity_snapshots ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
        data = [{"timestamp": r[0], "equity": float(r[1])} for r in rows]
        data.reverse()
        return data

    def summary_stats(self) -> Dict[str, float]:
        conn = self._connect()
        if not conn:
            return {}
        cur = conn.cursor()
        cur.execute(
            "SELECT profit_loss FROM trades WHERE profit_loss IS NOT NULL"
        )
        pnl_values = [float(r[0]) for r in cur.fetchall()]
        total_pnl = sum(pnl_values)
        wins = [p for p in pnl_values if p > 0]
        win_rate = round(len(wins) / len(pnl_values) * 100, 2) if pnl_values else 0.0
        avg_return = round(total_pnl / len(pnl_values), 4) if pnl_values else 0.0
        conn.close()
        return {
            "total_pnl": round(total_pnl, 4),
            "win_rate": win_rate,
            "avg_return": avg_return,
        }
