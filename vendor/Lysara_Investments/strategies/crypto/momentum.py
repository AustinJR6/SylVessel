# strategies/crypto/momentum.py

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np

from risk.risk import DynamicRisk
from services.daemon_state import get_state
from services.event_risk_service import EventRiskService
from services.exposure_service import ExposureService
from services.position_sizing_service import PositionSizingService
from utils.helpers import parse_price
from services.ai_strategist import get_ai_trade_decisions_batch
from strategies.base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    def __init__(self, api, risk, config, db, symbol_list, sentiment_source=None, ai_symbols=None, position_registry=None):
        super().__init__(api, risk, config, db, symbol_list)
        self.sentiment_source = sentiment_source
        self.ai_symbols = set(ai_symbols or [])
        self.position_registry = position_registry
        self.interval = int(config.get("market_poll_interval_seconds", 15))
        self.ai_min_confidence = float(config.get("ai_min_confidence", 0.7))
        self.ai_review_interval_seconds = int(config.get("ai_review_interval_seconds", 300))
        self.ai_decision_cache_ttl_seconds = int(
            config.get("ai_decision_cache_ttl_seconds", max(self.ai_review_interval_seconds * 3, 1800))
        )
        self.trade_cooldown_seconds = int(config.get("trade_cooldown_seconds", 180))
        self.stop_loss_pct = float(config.get("stop_loss_pct", 0.02))
        self.take_profit_pct = float(config.get("take_profit_pct", 0.04))
        self.max_position_value_pct = float(config.get("max_position_value_pct", 0.2))
        self.last_trade_at: dict[str, datetime] = {}
        self.last_ai_review_at: dict[str, datetime] = {}
        self.open_trade_plan: dict[str, dict] = {}
        self.event_risk_signatures: dict[str, str] = {}
        self.state = get_state()
        self.event_risk_service = EventRiskService(config)
        self.exposure_service = ExposureService(self.state)
        self.position_sizing_service = PositionSizingService(self.state, db=self.db, exposure_service=self.exposure_service)
        self.dynamic_risk = DynamicRisk(risk,
                                       config.get("atr_period", 14),
                                       config.get("volatility_multiplier", 3))

    async def run(self):
        while True:
            pending_reviews: list[tuple[str, float, float, dict, dict]] = []
            for symbol in self.symbols:
                try:
                    if not self.trading_enabled(symbol):
                        continue
                    data = await self.api.fetch_market_price(symbol)
                    price = parse_price(data)
                    if price <= 0:
                        self.record_decision(symbol, "observe", "blocked", ["missing_market_price"], market="crypto")
                        continue
                    from services.price_trigger_service import get_trigger_service
                    _trigger_svc = get_trigger_service()
                    if _trigger_svc:
                        _trigger_svc.update_price(symbol, price)
                    self.price_history[symbol].append(price)

                    if len(self.price_history[symbol]) > 100:
                        self.price_history[symbol] = self.price_history[symbol][-100:]

                    current_qty = await self._current_position_qty(symbol)
                    event_risk = self.event_risk_service.get_symbol_risk(symbol)
                    if current_qty > 0:
                        if await self._manage_open_position(symbol, price, current_qty):
                            continue
                        if await self._manage_event_risk(symbol, price, current_qty, event_risk):
                            continue

                    if not self._ai_review_due(symbol):
                        continue
                    sentiment = await self.get_sentiment(symbol)
                    context = self._build_context(symbol, price, sentiment, event_risk)
                    pending_reviews.append((symbol, price, current_qty, event_risk, context))

                except Exception as e:
                    logging.error(f"[Momentum] Error on {symbol}: {e}")

            if pending_reviews:
                decisions = await get_ai_trade_decisions_batch(
                    [context for _, _, _, _, context in pending_reviews],
                    cache_key_prefix="crypto:momentum",
                    min_interval_seconds=self.ai_review_interval_seconds,
                    max_cache_age_seconds=self.ai_decision_cache_ttl_seconds,
                )
                for symbol, price, current_qty, event_risk, _ in pending_reviews:
                    self.last_ai_review_at[symbol] = datetime.utcnow()
                    decision = decisions.get(symbol, {"action": "hold", "confidence": 0.5, "reason": "no_ai_decision"})
                    decision_action = str(decision.get("action") or "").strip().lower()
                    decision_confidence = float(decision.get("confidence", 0.0) or 0.0)

                    if decision_action in ["buy", "sell"]:
                        if decision_confidence < self.ai_min_confidence and not self.state.override_active("confidence_minimum"):
                            logging.info("Momentum: confidence threshold blocked %s on %s.", decision_action, symbol)
                            self.record_decision(symbol, decision_action, "blocked", ["confidence_below_threshold"], confidence=decision_confidence, market="crypto")
                            continue
                        if decision_action == "buy" and current_qty > 0:
                            logging.info("Momentum: existing long position on %s; skipping duplicate buy.", symbol)
                            self.record_decision(symbol, decision_action, "blocked", ["existing_long_position"], confidence=decision_confidence, market="crypto")
                            continue
                        if decision_action == "buy" and event_risk.get("block_new_positions") and not self.state.override_active("event_risk_warning"):
                            logging.info("Momentum: event risk blocked new %s on %s (%s).", decision_action, symbol, event_risk.get("action"))
                            self.record_decision(symbol, decision_action, "blocked", ["event_risk_blocked"], confidence=decision_confidence, market="crypto")
                            continue
                        if not self._trade_cooldown_elapsed(symbol) and not self.state.override_active("trade_cooldown"):
                            logging.info("Momentum: cooldown active for %s; skipping trade.", symbol)
                            self.record_decision(symbol, decision_action, "blocked", ["trade_cooldown"], confidence=decision_confidence, market="crypto")
                            continue
                        await self.enter_trade(
                            symbol,
                            price,
                            decision_action,
                            decision_confidence,
                            decision.get("reason", "ai"),
                        )
                    else:
                        logging.info(f"AI decision skipped: {decision}")
                        self.record_decision(symbol, "observe", "skipped", [str(decision.get("reason") or "hold")], confidence=decision_confidence, market="crypto")

            await asyncio.sleep(self.interval)

    async def get_sentiment(self, symbol: str) -> float:
        if not self.sentiment_source:
            return 0.0
        scores = self.sentiment_source.sentiment_scores
        symbol_key = str(symbol or "").upper()
        symbol_entry = ((scores.get("symbols") or {}) if isinstance(scores.get("symbols"), dict) else {}).get(symbol_key)
        if isinstance(symbol_entry, dict):
            composite = symbol_entry.get("composite") or {}
            try:
                return float(composite.get("score") or 0.0)
            except Exception:
                return 0.0

        reddit_data = scores.get("reddit", {})
        news_data = scores.get("newsapi", {})
        reddit_score = 0.0
        news_score = 0.0

        if isinstance(reddit_data, dict):
            if symbol_key in reddit_data and isinstance(reddit_data.get(symbol_key), dict):
                reddit_score = float((reddit_data.get(symbol_key) or {}).get("score") or 0.0)
            elif reddit_data:
                entries = [entry for entry in reddit_data.values() if isinstance(entry, dict)]
                if entries:
                    reddit_score = sum(float(entry.get("score") or 0.0) for entry in entries) / len(entries)

        if isinstance(news_data, dict):
            if symbol_key in news_data and isinstance(news_data.get(symbol_key), dict):
                news_score = float((news_data.get(symbol_key) or {}).get("score") or 0.0)
            elif "score" in news_data:
                news_score = float(news_data.get("score") or 0.0)

        return (reddit_score + news_score) / 2

    def _build_context(self, symbol: str, price: float, sentiment: float, event_risk: dict | None = None) -> dict:
        prices = self.price_history[symbol]
        vol = float(np.std(np.diff(prices[-10:]))) if len(prices) > 2 else 0.0
        trend = "sideways"
        if len(prices) >= 5:
            x = np.arange(5)
            y = np.array(prices[-5:])
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0:
                trend = "uptrend"
            elif slope < 0:
                trend = "downtrend"
        pos_qty = 0.0
        if getattr(self.api, "portfolio", None):
            pos_qty = self.api.portfolio.open_positions.get(symbol, 0.0)
        status = "flat"
        if pos_qty > 0:
            status = "long"
        elif pos_qty < 0:
            status = "short"
        support = min(prices[-20:]) if len(prices) >= 1 else price
        resistance = max(prices[-20:]) if len(prices) >= 1 else price
        return {
            "symbol": symbol,
            "price": price,
            "volatility": round(vol, 6),
            "sentiment": sentiment,
            "event_risk_score": float((event_risk or {}).get("risk_score") or 0.0),
            "event_risk_action": str((event_risk or {}).get("action") or "normal"),
            "event_risk_reasons": list((event_risk or {}).get("reasons") or []),
            "position_status": status,
            "recent_trend": trend,
            "support": support,
            "resistance": resistance,
            "drawdown": self.risk.daily_loss,
            "loss_streak": self.risk.consec_losses,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def enter_trade(
        self,
        symbol: str,
        price: float,
        action: str,
        confidence: float,
        reason: str,
        *,
        desired_qty: float | None = None,
    ) -> bool:
        if not self.trading_enabled(symbol):
            logging.info("Strategy or symbol disabled. Trade skipped.")
            self.record_decision(symbol, action, "blocked", ["strategy_or_symbol_disabled"], confidence=confidence, market="crypto")
            return False
        if action == "buy":
            allowed, registry_reasons = self.registry_allows(symbol, action)
            if not allowed:
                self.record_decision(symbol, action, "blocked", registry_reasons, confidence=confidence, market="crypto")
                return False
        if not self.config.get("ENABLE_AI_TRADE_EXECUTION", False):
            logging.info("AI trade execution disabled. Trade skipped.")
            self.record_decision(symbol, action, "blocked", ["ai_trade_execution_disabled"], confidence=confidence, market="crypto")
            return False
        if not await self.risk.check_daily_loss():
            logging.warning("Daily loss limit reached. Trade blocked.")
            self.record_decision(symbol, action, "blocked", ["daily_loss_limit"], confidence=confidence, market="crypto")
            return False
        current_qty = await self._current_position_qty(symbol)
        event_risk = self.event_risk_service.get_symbol_risk(symbol)
        if action == "buy" and event_risk.get("block_new_positions") and not self.state.override_active("event_risk_warning"):
            logging.info("Momentum: event risk blocked buy on %s (%s).", symbol, event_risk.get("action"))
            self.record_decision(symbol, action, "blocked", ["event_risk_blocked"], confidence=confidence, market="crypto")
            return False
        if action == "sell" and current_qty <= 0:
            logging.info("Momentum: no open position on %s; sell skipped.", symbol)
            self.record_decision(symbol, action, "blocked", ["no_open_position"], confidence=confidence, market="crypto")
            return False

        sizing = self.position_sizing_service.compute_order_size(
            market="crypto",
            symbol=symbol,
            side=action,
            price=price,
            confidence=confidence,
            risk_manager=self.risk,
            price_history=self.price_history[symbol],
            desired_qty=desired_qty,
            current_position_qty=current_qty,
        )
        qty = float(sizing.get("quantity") or 0.0)
        if symbol in self.ai_symbols and desired_qty is None:
            qty *= 0.5
            reason = f"AI_DISCOVERED | {reason}"
        qty = await self._normalize_order_quantity(symbol, price, qty, action, current_qty=current_qty)
        if qty <= 0:
            logging.warning("Momentum: invalid position size for %s (%s).", symbol, ",".join(sizing.get("capped_by") or []) or "no_capacity")
            self.record_decision(symbol, action, "blocked", ["invalid_position_size"], confidence=confidence, market="crypto")
            return False

        stops = self.dynamic_risk.stop_levels(price, action, self.price_history[symbol])
        if stops.rr < 1.0:
            logging.info(f"Trade skipped on {symbol} due to RR {stops.rr:.2f}")
            self.record_decision(symbol, action, "blocked", ["risk_reward_too_low"], confidence=confidence, market="crypto")
            return False

        if not self.config.get("simulation_mode", True) and not self.config.get("LIVE_TRADING_ENABLED", True):
            logging.info("Live trading disabled. Trade skipped.")
            self.record_decision(symbol, action, "blocked", ["live_trading_disabled"], confidence=confidence, market="crypto")
            return False

        await self.api.place_order(
            symbol=symbol,
            side=action,
            qty=qty,
            order_type="MARKET",
            price=price,
            confidence=confidence,
        )

        realized_pnl = None
        if action == "sell":
            plan = self.open_trade_plan.get(symbol) or {}
            entry_price = float(plan.get("entry_price") or 0.0)
            if entry_price > 0:
                realized_pnl = round((price - entry_price) * qty, 4)
                if realized_pnl < 0:
                    self.risk.record_loss(realized_pnl)
                else:
                    self.risk.reset_streak()

        self.db.log_trade(
            symbol=symbol,
            side=action,
            quantity=qty,
            price=price,
            profit_loss=realized_pnl,
            reason=reason,
            market="crypto",
        )
        self.last_trade_at[symbol] = datetime.utcnow()
        self._record_trade_plan(symbol, action, price, qty)

        logging.info(
            f"{action.upper()} {symbol} @ {price} conf={confidence} RR={stops.rr:.2f} details={reason}"
        )
        self.record_decision(symbol, action, "executed", [reason], confidence=confidence, market="crypto")
        return True

    async def _current_position_qty(self, symbol: str) -> float:
        holdings = await self.api.fetch_holdings()
        if not isinstance(holdings, dict):
            return 0.0
        asset_code = symbol.split("-", 1)[0].upper()
        return float(holdings.get(symbol, holdings.get(asset_code, 0.0)) or 0.0)

    async def _account_buying_power(self) -> float:
        info = await self.api.fetch_account_info()
        if not isinstance(info, dict):
            return 0.0
        return float(info.get("buying_power", info.get("balance", info.get("portfolio_value", 0.0))) or 0.0)

    async def _normalize_order_quantity(
        self,
        symbol: str,
        price: float,
        qty: float,
        action: str,
        *,
        current_qty: float = 0.0,
    ) -> float:
        if qty <= 0 or price <= 0:
            return 0.0
        precision_cfg = self.config.get("trade_precision", {}) or {}
        min_trade_cfg = self.config.get("min_trade_size", {}) or {}
        precision = int(precision_cfg.get(symbol, 6))
        min_qty = float(min_trade_cfg.get(symbol, 0.0) or 0.0)

        if action == "buy":
            buying_power = await self._account_buying_power()
            max_position_value = max(buying_power * self.max_position_value_pct, min_qty * price)
            affordable_qty = buying_power / price if price > 0 else 0.0
            capped_qty = min(qty, affordable_qty, max_position_value / price if price > 0 else 0.0)
        else:
            capped_qty = min(qty, current_qty)

        capped_qty = round(float(capped_qty or 0.0), precision)
        if capped_qty <= 0:
            return 0.0
        if min_qty > 0 and capped_qty < min_qty:
            return 0.0
        return capped_qty

    def _trade_cooldown_elapsed(self, symbol: str) -> bool:
        last_trade = self.last_trade_at.get(symbol)
        if not last_trade:
            return True
        return datetime.utcnow() - last_trade >= timedelta(seconds=self.trade_cooldown_seconds)

    def _ai_review_due(self, symbol: str) -> bool:
        last_review = self.last_ai_review_at.get(symbol)
        if not last_review:
            return True
        return datetime.utcnow() - last_review >= timedelta(seconds=self.ai_review_interval_seconds)

    def _record_trade_plan(self, symbol: str, action: str, entry_price: float, quantity: float) -> None:
        if action == "buy":
            stop_price = float(entry_price) * (1 - self.stop_loss_pct)
            target_price = float(entry_price) * (1 + self.take_profit_pct)
            self.open_trade_plan[symbol] = {
                "entry_price": float(entry_price),
                "quantity": float(quantity),
                "stop_price": stop_price,
                "target_price": target_price,
            }
            from services.position_registry import RegisteredPosition, get_registry
            registry = self.position_registry or get_registry()
            registry.register(RegisteredPosition(
                symbol=symbol,
                strategy_name="MomentumStrategy",
                side="buy",
                entry_price=float(entry_price),
                quantity=float(quantity),
                stop_price=stop_price,
                target_price=target_price,
            ))
        elif action == "sell":
            self.open_trade_plan.pop(symbol, None)
            from services.position_registry import get_registry
            registry = self.position_registry or get_registry()
            registry.release(symbol, "MomentumStrategy")

    async def _manage_open_position(self, symbol: str, price: float, current_qty: float) -> bool:
        plan = self.open_trade_plan.get(symbol)
        if not plan:
            self.open_trade_plan[symbol] = {
                "entry_price": float(price),
                "quantity": float(current_qty),
                "stop_price": float(price) * (1 - self.stop_loss_pct),
                "target_price": float(price) * (1 + self.take_profit_pct),
            }
            return False
        if price >= float(plan["target_price"]):
            await self.enter_trade(symbol, price, "sell", 1.0, "take_profit")
            return True
        if price <= float(plan["stop_price"]):
            await self.enter_trade(symbol, price, "sell", 1.0, "stop_loss")
            return True
        return False

    async def _manage_event_risk(self, symbol: str, price: float, current_qty: float, event_risk: dict) -> bool:
        action = str((event_risk or {}).get("action") or "normal")
        if action != "reduce_positions" or current_qty <= 0 or self.state.override_active("event_risk_warning"):
            if action == "normal":
                self.event_risk_signatures.pop(symbol, None)
            return False
        signature = str((event_risk or {}).get("primary_event_key") or "").strip()
        if not signature or self.event_risk_signatures.get(symbol) == signature:
            return False
        reduction_pct = float((event_risk or {}).get("reduce_position_pct") or self.config.get("event_risk_reduction_factor", 0.5) or 0.5)
        desired_qty = max(current_qty * reduction_pct, 0.0)
        if desired_qty <= 0:
            return False
        reduced = await self.enter_trade(
            symbol,
            price,
            "sell",
            1.0,
            f"event_risk_reduce | {'; '.join((event_risk or {}).get('reasons') or [])}",
            desired_qty=desired_qty,
        )
        if reduced:
            self.event_risk_signatures[symbol] = signature
        return reduced
