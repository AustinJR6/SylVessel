"""Adaptive strategy routing and allocation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RouterConfig:
    """Configuration parameters for the meta-strategy router."""

    max_weight: float = 0.60
    min_weight: float = 0.0
    score_floor: float = -5.0
    score_cap: float = 5.0


class MetaStrategyRouter:
    """Rank and allocate among active strategies based on efficacy and regime.

    Inputs:
    - Real-time metrics (for example latency, recent pnl, error rate)
    - Recent performance statistics (for example sharpe, win rate, drawdown)
    - Regime signal (trend/range/high_volatility)
    """

    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()
        self.performance_stats: dict[str, dict[str, Any]] = {}

    def update_performance(
        self,
        strategy_name: str,
        sharpe_ratio: float,
        win_rate: float,
        total_return: float,
        max_drawdown: float = 0.0,
    ) -> None:
        self.performance_stats[str(strategy_name or "").strip() or "unknown"] = {
            "sharpe_ratio": float(sharpe_ratio or 0.0),
            "win_rate": float(win_rate or 0.0),
            "total_return": float(total_return or 0.0),
            "max_drawdown": float(max_drawdown or 0.0),
        }

    @staticmethod
    def _to_df(stats: Mapping[str, Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(stats, pd.DataFrame):
            df = stats.copy()
            if "strategy" in df.columns:
                df = df.set_index("strategy")
            return df
        df = pd.DataFrame.from_dict(stats, orient="index")
        df.index.name = "strategy"
        return df

    @staticmethod
    def _z(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / std

    @staticmethod
    def _regime_multiplier(strategy_name: str, regime: str) -> float:
        name = strategy_name.lower()
        regime = (regime or "").lower()
        if regime == "trend":
            if "momentum" in name or "breakout" in name:
                return 1.20
            if "mean_reversion" in name or "pairs" in name:
                return 0.85
        if regime == "range":
            if "mean_reversion" in name or "pairs" in name:
                return 1.20
            if "momentum" in name or "breakout" in name:
                return 0.85
        if regime in {"high_volatility", "high_vol"}:
            if "scalp" in name:
                return 1.15
            if "momentum" in name:
                return 1.05
            if "earnings" in name:
                return 0.80
        return 1.0

    def _compute_scores(
        self,
        perf_df: pd.DataFrame,
        realtime_metrics: Mapping[str, Mapping[str, Any]] | None = None,
        regime: str = "trend",
    ) -> pd.Series:
        df = perf_df.copy()
        for col in ("sharpe_ratio", "total_return", "win_rate", "max_drawdown"):
            if col not in df.columns:
                df[col] = 0.0

        # Linear ranking model over normalized factors.
        score = (
            0.40 * self._z(pd.to_numeric(df["sharpe_ratio"], errors="coerce").fillna(0.0))
            + 0.30 * self._z(pd.to_numeric(df["total_return"], errors="coerce").fillna(0.0))
            + 0.20 * self._z(pd.to_numeric(df["win_rate"], errors="coerce").fillna(0.0))
            - 0.30 * self._z(pd.to_numeric(df["max_drawdown"], errors="coerce").abs().fillna(0.0))
        )

        if realtime_metrics:
            rt = pd.DataFrame.from_dict(realtime_metrics, orient="index")
            rt_recent = pd.to_numeric(rt.get("recent_pnl", 0.0), errors="coerce").fillna(0.0)
            rt_errors = pd.to_numeric(rt.get("error_rate", 0.0), errors="coerce").fillna(0.0)
            score = score.add(0.20 * self._z(rt_recent.reindex(score.index).fillna(0.0)), fill_value=0.0)
            score = score.add(-0.15 * self._z(rt_errors.reindex(score.index).fillna(0.0)), fill_value=0.0)

        for strat in score.index:
            score.loc[strat] *= self._regime_multiplier(strat, regime)
        return score.clip(self.config.score_floor, self.config.score_cap)

    def route(
        self,
        performance_stats: Mapping[str, Mapping[str, Any]] | pd.DataFrame,
        regime_signal: str,
        realtime_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[dict[str, float | str]]:
        """Return prioritized strategy allocation percentages."""

        perf_df = self._to_df(performance_stats)
        if perf_df.empty:
            return []

        scores = self._compute_scores(perf_df, realtime_metrics=realtime_metrics, regime=regime_signal)
        # Softmax for non-negative, normalized allocation.
        exp_scores = np.exp(scores - scores.max())
        weights = pd.Series(exp_scores / exp_scores.sum(), index=scores.index)

        # Constrained projection (simple cap + renormalization loop).
        max_w = self.config.max_weight
        min_w = self.config.min_weight
        weights = weights.clip(lower=min_w, upper=max_w)
        total = weights.sum()
        if total > 0:
            weights = weights / total

        rows: list[dict[str, float | str]] = []
        for strategy, weight in weights.sort_values(ascending=False).items():
            rows.append(
                {
                    "strategy": strategy,
                    "score": float(scores.loc[strategy]),
                    "allocation_pct": float(weight * 100.0),
                }
            )
        return rows

    def current_route(
        self,
        regime_signal: str,
        realtime_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[dict[str, float | str]]:
        return self.route(self.performance_stats, regime_signal=regime_signal, realtime_metrics=realtime_metrics)


if __name__ == "__main__":
    example_perf = {
        "crypto_momentum": {"sharpe_ratio": 1.2, "total_return": 0.18, "win_rate": 0.55, "max_drawdown": -0.08},
        "crypto_mean_reversion": {"sharpe_ratio": 0.9, "total_return": 0.11, "win_rate": 0.61, "max_drawdown": -0.05},
        "micro_scalping": {"sharpe_ratio": 0.7, "total_return": 0.09, "win_rate": 0.52, "max_drawdown": -0.06},
    }
    router = MetaStrategyRouter()
    ranked = router.route(example_perf, regime_signal="trend")
    for row in ranked:
        print(row)
