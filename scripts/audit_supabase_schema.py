from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import psycopg2


ROOT = Path(__file__).resolve().parents[1]
TARGET_FILES = [
    ROOT / "server.py",
    ROOT / "memory" / "memory_manager.py",
    ROOT / "memory" / "lysara_memory_manager.py",
]

CREATE_TABLE_RE = re.compile(
    r"CREATE TABLE IF NOT EXISTS\s+((?:[a-zA-Z_][a-zA-Z0-9_]*\.)?[a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)
ALTER_COLUMN_RE = re.compile(
    r"ALTER TABLE\s+((?:[a-zA-Z_][a-zA-Z0-9_]*\.)?[a-zA-Z_][a-zA-Z0-9_]*)\s+ADD COLUMN IF NOT EXISTS\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)
CREATE_FUNCTION_RE = re.compile(
    r"CREATE OR REPLACE FUNCTION\s+((?:[a-zA-Z_][a-zA-Z0-9_]*\.)?[a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)
CREATE_EXTENSION_RE = re.compile(
    r"CREATE EXTENSION IF NOT EXISTS\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)


def _split_qualified(name: str) -> tuple[str, str]:
    raw = (name or "").strip().lower()
    if "." in raw:
        schema, table = raw.split(".", 1)
        return schema, table
    return "public", raw


def _load_local_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _collect_expected_schema() -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[str]], set[tuple[str, str]], set[str]]:
    expected_tables: set[tuple[str, str]] = set()
    expected_columns: dict[tuple[str, str], set[str]] = {}
    expected_functions: set[tuple[str, str]] = set()
    expected_extensions: set[str] = set()

    for path in TARGET_FILES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in CREATE_TABLE_RE.finditer(text):
            expected_tables.add(_split_qualified(match.group(1)))
        for match in ALTER_COLUMN_RE.finditer(text):
            expected_columns.setdefault(_split_qualified(match.group(1)), set()).add(match.group(2).lower())
        for match in CREATE_FUNCTION_RE.finditer(text):
            expected_functions.add(_split_qualified(match.group(1)))
        for match in CREATE_EXTENSION_RE.finditer(text):
            expected_extensions.add(match.group(1).lower())

    return expected_tables, expected_columns, expected_functions, expected_extensions


def _fetch_actual_schema(db_url: str) -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[str]], set[tuple[str, str]], set[str]]:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            """
        )
        tables = {(row[0].lower(), row[1].lower()) for row in cur.fetchall()}

        cur.execute(
            """
            SELECT table_schema, table_name, column_name
            FROM information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            """
        )
        columns: dict[tuple[str, str], set[str]] = {}
        for schema, table, column in cur.fetchall():
            columns.setdefault((schema.lower(), table.lower()), set()).add(column.lower())

        cur.execute(
            """
            SELECT n.nspname, p.proname
            FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname NOT IN ('information_schema', 'pg_catalog')
            """
        )
        functions = {(row[0].lower(), row[1].lower()) for row in cur.fetchall()}

        cur.execute("SELECT extname FROM pg_extension")
        extensions = {row[0].lower() for row in cur.fetchall()}
        return tables, columns, functions, extensions
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def main() -> int:
    _load_local_env()
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        print(json.dumps({"error": "SUPABASE_DB_URL missing"}, indent=2))
        return 2

    expected_tables, expected_columns, expected_functions, expected_extensions = _collect_expected_schema()
    actual_tables, actual_columns, actual_functions, actual_extensions = _fetch_actual_schema(db_url)

    missing_tables = sorted(expected_tables - actual_tables)
    missing_columns: dict[str, list[str]] = {}
    for key, columns in expected_columns.items():
        absent = sorted(columns - actual_columns.get(key, set()))
        if absent:
            missing_columns[f"{key[0]}.{key[1]}"] = absent
    missing_functions = sorted(expected_functions - actual_functions)
    missing_extensions = sorted(expected_extensions - actual_extensions)

    report = {
        "missing_tables": [f"{schema}.{table}" for schema, table in missing_tables],
        "missing_columns": missing_columns,
        "missing_functions": [f"{schema}.{name}" for schema, name in missing_functions],
        "missing_extensions": missing_extensions,
    }
    print(json.dumps(report, indent=2))
    if report["missing_tables"] or report["missing_columns"] or report["missing_functions"] or report["missing_extensions"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
