from __future__ import annotations

from typing import Dict

from memory.supabase_client import pooled_cursor


class Reporting:
    def summary(self) -> Dict[str, object]:
        out: Dict[str, object] = {}

        try:
            with pooled_cursor(commit=False) as cur:
                cur.execute("SELECT COUNT(*) FROM memories")
                out["memories_total"] = int(cur.fetchone()[0] or 0)
        except Exception:
            out["memories_total"] = 0

        try:
            with pooled_cursor(commit=False) as cur:
                cur.execute("SELECT COUNT(*) FROM prospects")
                out["prospects_total"] = int(cur.fetchone()[0] or 0)
        except Exception:
            out["prospects_total"] = 0

        try:
            with pooled_cursor(commit=False) as cur:
                cur.execute("SELECT COUNT(*) FROM email_drafts WHERE status = 'draft'")
                out["drafts_pending"] = int(cur.fetchone()[0] or 0)
        except Exception:
            out["drafts_pending"] = 0

        return out
