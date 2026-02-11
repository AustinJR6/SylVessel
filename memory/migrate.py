"""
Sylana Vessel - Database Migration Tool
Safely applies schema changes to the database
"""

import sqlite3
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import config


def get_migration_files():
    """Get all migration SQL files in order"""
    migrations_dir = Path(__file__).parent / "migrations"
    if not migrations_dir.exists():
        print(f"⚠️  Migrations directory not found: {migrations_dir}")
        return []

    sql_files = sorted(migrations_dir.glob("*.sql"))
    return sql_files


def get_applied_migrations(conn):
    """Get list of already applied migrations"""
    cursor = conn.cursor()

    # Create migrations tracking table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_name TEXT UNIQUE NOT NULL,
            applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    cursor.execute("SELECT migration_name FROM schema_migrations")
    return {row[0] for row in cursor.fetchall()}


def apply_migration(conn, migration_file):
    """Apply a single migration file"""
    migration_name = migration_file.name

    print(f"📝 Applying migration: {migration_name}")

    try:
        # Read and execute migration SQL
        with open(migration_file, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        # Execute each statement (split by semicolon)
        cursor = conn.cursor()
        cursor.executescript(sql_script)

        # Record migration as applied
        cursor.execute(
            "INSERT INTO schema_migrations (migration_name) VALUES (?)",
            (migration_name,)
        )

        conn.commit()
        print(f"✅ Migration {migration_name} applied successfully")
        return True

    except Exception as e:
        print(f"❌ Error applying migration {migration_name}: {e}")
        conn.rollback()
        return False


def run_migrations(db_path=None):
    """Run all pending migrations"""
    if db_path is None:
        db_path = config.DB_PATH

    print("=" * 60)
    print("  SYLANA VESSEL - DATABASE MIGRATION")
    print("=" * 60)
    print(f"\n📂 Database: {db_path}\n")

    # Ensure database directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        print("✅ Connected to database\n")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False

    try:
        # Get migrations
        migration_files = get_migration_files()
        if not migration_files:
            print("⚠️  No migration files found")
            return True

        applied = get_applied_migrations(conn)
        pending = [m for m in migration_files if m.name not in applied]

        if not pending:
            print("✅ All migrations already applied. Database is up to date.\n")
            return True

        print(f"📋 Found {len(pending)} pending migration(s):\n")
        for migration in pending:
            print(f"   - {migration.name}")
        print()

        # Apply each pending migration
        success_count = 0
        for migration_file in pending:
            if apply_migration(conn, migration_file):
                success_count += 1
            else:
                print("\n⚠️  Migration failed. Stopping here.")
                break

        print()
        print("=" * 60)
        print(f"✅ Applied {success_count}/{len(pending)} migration(s)")
        print("=" * 60)

        return success_count == len(pending)

    finally:
        conn.close()
        print("\n✅ Database connection closed")


def show_schema(db_path=None):
    """Display current database schema"""
    if db_path is None:
        db_path = config.DB_PATH

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("\n" + "=" * 60)
        print("  CURRENT DATABASE SCHEMA")
        print("=" * 60 + "\n")

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            print(f"\n📊 Table: {table_name}")
            print("-" * 60)

            # Get table info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            for col in columns:
                col_id, name, dtype, not_null, default, pk = col
                pk_marker = " [PK]" if pk else ""
                null_marker = " NOT NULL" if not_null else ""
                default_marker = f" DEFAULT {default}" if default else ""
                print(f"   {name}: {dtype}{pk_marker}{null_marker}{default_marker}")

            # Get indices for this table
            cursor.execute(f"PRAGMA index_list({table_name})")
            indices = cursor.fetchall()
            if indices:
                print(f"\n   Indices:")
                for idx in indices:
                    idx_name = idx[1]
                    print(f"      - {idx_name}")

        conn.close()
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"❌ Error displaying schema: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show_schema()
    else:
        success = run_migrations()
        print()
        if success:
            print("🎉 Migration complete! Showing updated schema...\n")
            show_schema()
        else:
            print("⚠️  Some migrations failed. Check errors above.")
            sys.exit(1)
