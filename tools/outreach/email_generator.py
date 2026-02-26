from __future__ import annotations

from typing import Dict


class EmailGenerator:
    def generate_email(self, company: str, contact: str, value_prop: str) -> Dict[str, str]:
        contact_name = contact or "there"
        company_name = company or "your team"
        subject = f"{company_name} + Manifest workflow fit"
        body = (
            f"Hi {contact_name},\\n\\n"
            f"Reaching out because {value_prop}. "
            "If it helps, I can share a 10-minute walkthrough tailored to your current install workflow.\\n\\n"
            "Best,\\nSylVessel"
        )
        return {"subject": subject, "body": body}
