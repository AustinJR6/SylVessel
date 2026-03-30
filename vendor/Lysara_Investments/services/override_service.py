from __future__ import annotations

from typing import Any


class OverrideService:
    def __init__(self, state, db=None):
        self.state = state
        self.db = db

    def status(self) -> dict[str, Any]:
        return self.state.get_override_status()

    def is_active(self, control_name: str | None = None) -> bool:
        return self.state.override_active(control_name)

    def activate(
        self,
        *,
        actor: str,
        reason: str,
        ttl_minutes: int | None = None,
        allowed_controls: list[str] | None = None,
    ) -> dict[str, Any]:
        ttl = int(ttl_minutes or self.state.config.get("override_ttl_minutes", 15))
        status = self.state.activate_override(
            actor=actor,
            reason=reason,
            ttl_minutes=ttl,
            allowed_controls=allowed_controls,
        )
        if self.db is not None:
            self.db.log_audit_event(
                actor=actor,
                event_type="override_activate",
                target="runtime_override",
                status="applied",
                details={"reason": reason, "ttl_minutes": ttl, "allowed_controls": status.get("allowed_controls") or []},
            )
            self.db.log_decision_journal(
                mode="direct_ops",
                action="override_activate",
                status="recorded",
                summary=f"Override activated by {actor}",
                market="crypto",
                symbol=None,
                details={"reason": reason, "ttl_minutes": ttl, "allowed_controls": status.get("allowed_controls") or []},
                trade_intent_id=None,
            )
        return status

    def clear(self, *, actor: str, reason: str = "") -> dict[str, Any]:
        status = self.state.clear_override(actor=actor, reason=reason)
        if self.db is not None:
            self.db.log_audit_event(
                actor=actor,
                event_type="override_clear",
                target="runtime_override",
                status="applied",
                details={"reason": reason},
            )
            self.db.log_decision_journal(
                mode="direct_ops",
                action="override_clear",
                status="recorded",
                summary=f"Override cleared by {actor}",
                market="crypto",
                symbol=None,
                details={"reason": reason},
                trade_intent_id=None,
            )
        return status
