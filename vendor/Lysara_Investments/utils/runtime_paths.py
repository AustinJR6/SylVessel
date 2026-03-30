from __future__ import annotations

import os
from pathlib import Path


def get_runtime_dir() -> Path:
    raw = str(os.getenv("LYSARA_RUNTIME_DIR", "runtime_data") or "runtime_data").strip()
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_path(*parts: str) -> Path:
    path = get_runtime_dir().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def env_or_runtime_path(env_name: str, *parts: str) -> Path:
    raw = str(os.getenv(env_name, "") or "").strip()
    if raw:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return runtime_path(*parts)

