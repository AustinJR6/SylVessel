"""Filtering helpers for sentiment sources."""
from __future__ import annotations

from typing import List, Set, Dict


def filter_accounts(items: List[Dict], whitelist: Set[str] | None = None, blacklist: Set[str] | None = None) -> List[Dict]:
    """Remove spam/bot accounts based on follower/karma counts."""
    whitelist = whitelist or set()
    blacklist = blacklist or set()
    filtered: List[Dict] = []
    for item in items:
        user = item.get("user", "")
        followers = item.get("followers", 0)
        karma = item.get("karma", 0)
        if user in blacklist:
            continue
        if user in whitelist or followers >= 20 or karma >= 30:
            filtered.append(item)
    return filtered

