"""Capital allocation engine for strategy portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationConfig:
    """Risk and exposure constraints."""

    max_strategy_risk_pct: float = 0.05
    max_portfolio_risk_pct: float = 0.20
    corr_threshold: float = 0.80
    overlap_penalty: float = 0.50


class CapitalAllocator:
    """Allocate strategy capital with risk caps and correlation filtering."""

    def __init__(self, config: AllocationConfig | None = None):
        self.config = config or AllocationConfig()

    def _as_df(self, strategies: list[Mapping[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(strategies)
        if "strategy" not in df.columns:
            raise ValueError("Each strategy payload must include 'strategy'.")
        if "score" not in df.columns and "allocation_pct" not in df.columns:
            raise ValueError("Each strategy payload must include 'score' or 'allocation_pct'.")
        if "score" not in df.columns:
            df["score"] = pd.to_numeric(df["allocation_pct"], errors="coerce").fillna(0.0)
        else:
            df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
        return df

    def _apply_correlation_filter(
        self,
        df: pd.DataFrame,
        correlation_matrix: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if correlation_matrix is None or correlation_matrix.empty:
            return df

        out = df.copy()
        corr = correlation_matrix.copy()
        active = set(out["strategy"].tolist())
        corr = corr.reindex(index=active, columns=active).fillna(0.0)

        by_score = out.sort_values("score", ascending=False)["strategy"].tolist()
        suppressed: set[str] = set()
        for i, s1 in enumerate(by_score):
            if s1 in suppressed:
                continue
            for s2 in by_score[i + 1 :]:
                if s2 in suppressed:
                    continue
                c = float(corr.loc[s1, s2]) if s1 in corr.index and s2 in corr.columns else 0.0
                if abs(c) >= self.config.corr_threshold:
                    suppressed.add(s2)
        if suppressed:
            out.loc[out["strategy"].isin(suppressed), "score"] *= self.config.overlap_penalty
        return out

    def allocate(
        self,
        total_capital: float,
        strategy_scores: list[Mapping[str, Any]],
        correlation_matrix: pd.DataFrame | None = None,
        overlap_groups: Mapping[str, str] | None = None,
    ) -> list[dict[str, float | str]]:
        """Return position sizing decisions by strategy."""

        if total_capital <= 0:
            return []

        df = self._as_df(strategy_scores)
        df = self._apply_correlation_filter(df, correlation_matrix)

        # Softmax weights from scores, then apply hard caps.
        scores = df["score"].to_numpy(dtype=float)
        exp_s = np.exp(scores - np.max(scores))
        raw_w = exp_s / exp_s.sum() if exp_s.sum() > 0 else np.zeros_like(exp_s)
        df["weight"] = raw_w

        max_w = self.config.max_strategy_risk_pct / max(self.config.max_portfolio_risk_pct, 1e-9)
        df["weight"] = df["weight"].clip(upper=max_w)
        if df["weight"].sum() > 0:
            df["weight"] = df["weight"] / df["weight"].sum()

        # Apply overlap caps by group (for example same asset family).
        if overlap_groups:
            df["group"] = df["strategy"].map(overlap_groups).fillna(df["strategy"])
            group_weight = df.groupby("group")["weight"].sum().to_dict()
            for idx, row in df.iterrows():
                gw = group_weight.get(row["group"], 0.0)
                if gw > 0.40:
                    scale = 0.40 / gw
                    df.at[idx, "weight"] = row["weight"] * scale
            if df["weight"].sum() > 0:
                df["weight"] = df["weight"] / df["weight"].sum()

        portfolio_risk_budget = total_capital * self.config.max_portfolio_risk_pct
        rows: list[dict[str, float | str]] = []
        for _, row in df.sort_values("weight", ascending=False).iterrows():
            strat_risk_budget = min(
                portfolio_risk_budget * float(row["weight"]),
                total_capital * self.config.max_strategy_risk_pct,
            )
            capital_alloc = float(total_capital * float(row["weight"]))
            rows.append(
                {
                    "strategy": str(row["strategy"]),
                    "allocation_pct": float(round(row["weight"] * 100.0, 4)),
                    "capital_allocated": float(round(capital_alloc, 2)),
                    "risk_budget": float(round(strat_risk_budget, 2)),
                }
            )
        return rows


if __name__ == "__main__":
    allocator = CapitalAllocator()
    decisions = allocator.allocate(
        total_capital=100_000,
        strategy_scores=[
            {"strategy": "crypto_momentum", "score": 1.2},
            {"strategy": "crypto_mean_reversion", "score": 0.8},
            {"strategy": "stock_momentum", "score": 1.0},
        ],
    )
    for row in decisions:
        print(row)
