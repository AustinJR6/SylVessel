from __future__ import annotations

from typing import Dict

from memory.supabase_client import pooled_cursor


class LeadCollector:
    def generate_leads(self, product: str = "manifest", limit: int = 5) -> Dict[str, object]:
        with pooled_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT prospect_id, company_name, contact_name, contact_title, email, status
                FROM prospects
                WHERE (%s = '' OR LOWER(product) = LOWER(%s))
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (product or "", product or "", max(1, min(int(limit), 50))),
            )
            rows = cur.fetchall()
        return {
            "leads": [
                {
                    "prospect_id": str(r[0]),
                    "company": r[1],
                    "contact": r[2],
                    "title": r[3],
                    "email": r[4],
                    "status": r[5],
                }
                for r in rows
            ]
        }
