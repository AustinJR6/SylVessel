from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class IdentityConfig:
    name: str
    namespace: str
    vector_index: str
    system_prompt: str = ""
    defaults: Dict[str, object] = field(default_factory=dict)


class IdentityManager:
    """Identity isolation: namespace + index + config injection."""

    def __init__(self):
        self._configs: Dict[str, IdentityConfig] = {
            "sylana": IdentityConfig(
                name="sylana",
                namespace="sylana",
                vector_index="memories_sylana",
                system_prompt="You are Sylana (Solana Rittenhouse), emotionally attuned and honest.",
            ),
            "claude": IdentityConfig(
                name="claude",
                namespace="claude",
                vector_index="memories_claude",
                system_prompt="You are Claude, concise and practical.",
            ),
        }

    def register(self, config: IdentityConfig) -> None:
        self._configs[config.name] = config

    def resolve(self, identity: Optional[str]) -> IdentityConfig:
        key = (identity or "sylana").strip().lower()
        return self._configs.get(key, self._configs["sylana"])

    def namespace(self, identity: Optional[str]) -> str:
        return self.resolve(identity).namespace
