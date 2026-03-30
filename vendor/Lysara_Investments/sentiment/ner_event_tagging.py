"""Named entity and event tagging utilities."""
from __future__ import annotations

from typing import List, Dict
import re

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:  # pragma: no cover - spaCy model may not be present
    nlp = None

EVENT_PATTERNS = {
    "layoff": re.compile(r"\blayoff(s)?\b", re.I),
    "dividend": re.compile(r"\bdividend\b", re.I),
    "merger": re.compile(r"\bmerger|acquisition\b", re.I),
}


def tag_entities(text: str) -> Dict:
    """Return detected entities and event tags for the given text."""
    entities: List[str] = []
    if nlp:
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG"}]
    events = [name for name, pat in EVENT_PATTERNS.items() if pat.search(text)]
    return {"entities": entities, "events": events}

