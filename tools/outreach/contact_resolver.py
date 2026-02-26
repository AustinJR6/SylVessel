from __future__ import annotations

from typing import Dict

from memory.supabase_client import pooled_cursor


class ContactResolver:
    def resolve_contact(self, company_name: str) -> Dict[str, object]:
        with pooled_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT prospect_id, company_name, contact_name, email, phone
                FROM prospects
                WHERE LOWER(company_name) = LOWER(%s)
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (company_name,),
            )
            row = cur.fetchone()
        if not row:
            return {"found": False}
        return {
            "found": True,
            "prospect_id": str(row[0]),
            "company": row[1],
            "contact": row[2],
            "email": row[3],
            "phone": row[4],
        }
