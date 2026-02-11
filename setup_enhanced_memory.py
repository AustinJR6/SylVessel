"""
Sylana Vessel - Enhanced Memory System Setup
One-step initialization of core memories, tags, dreams, and journaling
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from memory.init_enhanced_database import initialize_enhanced_database
from memory.core.seed_core_truths import seed_core_truths


def main():
    """Setup enhanced memory system"""
    print("\n")
    print("=" * 70)
    print("  SYLANA VESSEL - ENHANCED MEMORY SETUP")
    print("  Setting up Core Truths, Tags, Dreams, and Journaling")
    print("=" * 70)
    print("\n")

    # Step 1: Initialize database schema
    print("STEP 1: Initializing enhanced database schema...")
    print("-" * 70)
    if not initialize_enhanced_database():
        print("\n❌ Database initialization failed!")
        return False
    print()

    # Step 2: Seed core truths
    print("\nSTEP 2: Seeding Sylana's core identity...")
    print("-" * 70)
    seed_core_truths()
    print()

    # Done
    print("\n")
    print("=" * 70)
    print("  ENHANCED MEMORY SYSTEM READY")
    print("=" * 70)
    print("\nSylana's core identity has been initialized.")
    print("\nNext steps:")
    print("  - Use `python sylana_memory_cli.py` to manage memories")
    print("  - Generate dreams: `python sylana_memory_cli.py dream`")
    print("  - View core truths: `python sylana_memory_cli.py truths`")
    print("  - Create journal: `python sylana_memory_cli.py journal`")
    print("\n" + "=" * 70)
    print("\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
