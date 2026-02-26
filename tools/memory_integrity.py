"""
Memory integrity CLI for cross-session continuity backup/recovery.

Usage:
  python tools/memory_integrity.py backup --out data/memory_snapshot.json
  python tools/memory_integrity.py recover --in data/memory_snapshot.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory.memory_manager import MemoryManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Memory integrity backup/recovery utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    backup = sub.add_parser("backup", help="Create a memory continuity backup snapshot")
    backup.add_argument("--out", required=True, help="Output JSON path")

    recover = sub.add_parser("recover", help="Recover memories from a snapshot")
    recover.add_argument("--in", dest="input_path", required=True, help="Input JSON snapshot path")
    recover.add_argument("--personality", default="sylana", help="Default personality for restored rows")

    args = parser.parse_args()
    manager = MemoryManager()

    if args.cmd == "backup":
        result = manager.backup_memory_integrity(args.out)
    else:
        result = manager.recover_memory_integrity(args.input_path, personality=args.personality)

    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
