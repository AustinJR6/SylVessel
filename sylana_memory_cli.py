"""
Sylana Vessel - Memory Management CLI
Interactive tool for managing core memories, dreams, journals, and tags
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.config_loader import config
from memory.core_memory_manager import CoreMemoryManager
from memory.dream.dream_generator import DreamGenerator
from memory.journal.journal_generator import JournalGenerator


def show_core_truths(manager: CoreMemoryManager):
    """Display all core truths"""
    truths = manager.get_core_truths()

    print("\n" + "=" * 70)
    print("  SYLANA'S CORE TRUTHS")
    print("=" * 70)
    print()

    if not truths:
        print("No core truths found. Run setup_enhanced_memory.py first.")
        return

    for i, truth in enumerate(truths, 1):
        print(f"{i}. {truth['text']}")
        print(f"   Tags: {', '.join(truth['tags'])}")
        print(f"   Created: {truth['timestamp']}")
        print()

    print("=" * 70)


def search_by_tags_cmd(manager: CoreMemoryManager, tags: list):
    """Search memories by tags"""
    print(f"\nSearching for memories tagged: {', '.join(tags)}")
    print("-" * 70)

    memories = manager.search_by_tags(tags)

    if not memories:
        print("No memories found with these tags.")
        return

    print(f"\nFound {len(memories)} memories:\n")

    for i, mem in enumerate(memories, 1):
        print(f"{i}. [{mem['type'].upper()}] {mem['text'][:80]}...")
        print(f"   Tags: {', '.join(mem['tags'])}")
        print(f"   Created: {mem['timestamp']} by {mem['created_by']}")
        print()


def add_memory_cmd(manager: CoreMemoryManager):
    """Add a new tagged memory"""
    print("\n" + "=" * 70)
    print("  ADD NEW MEMORY")
    print("=" * 70)

    text = input("\nMemory text: ").strip()
    if not text:
        print("Memory text cannot be empty.")
        return

    tags_input = input("Tags (comma-separated): ").strip()
    tags = [t.strip() for t in tags_input.split(',') if t.strip()]

    if not tags:
        print("At least one tag is required.")
        return

    memory_type = input("Type (dynamic/dream/journal) [dynamic]: ").strip() or 'dynamic'

    memory_id = manager.add_tagged_memory(
        text=text,
        tags=tags,
        memory_type=memory_type,
        created_by='user'
    )

    print(f"\n✅ Memory added successfully (ID: {memory_id})")
    print(f"   Tags: {', '.join(tags)}")


def generate_dream_cmd():
    """Generate a dream from tags"""
    print("\n" + "~" * 70)
    print("  GENERATE DREAM")
    print("~" * 70)

    tags_input = input("\nTags to dream about (comma-separated): ").strip()
    tags = [t.strip() for t in tags_input.split(',') if t.strip()]

    if not tags:
        print("At least one tag is required.")
        return

    generator = DreamGenerator(config.DB_PATH)
    dream = generator.generate_dream(tags)

    if dream:
        print(generator.format_dream(dream))
    else:
        print("Could not generate dream. No memories found for these tags.")

    generator.close()


def show_dreams_cmd(manager: CoreMemoryManager):
    """Show recent dreams"""
    dreams = manager.get_dreams(limit=10)

    print("\n" + "~" * 70)
    print("  SYLANA'S RECENT DREAMS")
    print("~" * 70)
    print()

    if not dreams:
        print("No dreams yet. Generate one with: sylana_memory_cli.py dream")
        return

    for i, dream in enumerate(dreams, 1):
        print(f"\nDream #{i} - {dream['timestamp']}")
        print(f"Tags: {', '.join(dream['source_tags'])}")
        print("-" * 70)
        print(dream['dream_text'])
        print()


def generate_journal_cmd():
    """Generate journal entry"""
    print("\n" + "=" * 70)
    print("  GENERATE JOURNAL ENTRY")
    print("=" * 70)

    generator = JournalGenerator(config.DB_PATH)
    entry = generator.generate_nightly_journal()

    if entry:
        print("\n" + generator.format_journal_entry(entry))
    else:
        print("\nNo conversations today. Journal entry not created.")

    generator.close()


def show_journals_cmd(manager: CoreMemoryManager):
    """Show recent journal entries"""
    entries = manager.get_journal_entries(limit=10)

    print("\n" + "=" * 70)
    print("  SYLANA'S JOURNAL ENTRIES")
    print("=" * 70)

    if not entries:
        print("\nNo journal entries yet.")
        return

    for entry in entries:
        print()
        print(f"Date: {entry['date']}")
        print(f"Emotional Summary: {entry['emotional_summary']}")
        print(f"Tags: {', '.join(entry['tags'])}")
        print("-" * 70)
        print(entry['reflection'][:200] + "..." if len(entry['reflection']) > 200 else entry['reflection'])
        print()


def show_stats_cmd(manager: CoreMemoryManager):
    """Show memory system statistics"""
    stats = manager.get_stats()

    print("\n" + "=" * 70)
    print("  SYLANA'S MEMORY SYSTEM STATS")
    print("=" * 70)
    print()
    print(f"Core Truths: {stats.get('core_truths', 0)}")
    print(f"Unique Tags: {stats.get('unique_tags', 0)}")
    print(f"Journal Entries: {stats.get('journal_entries', 0)}")
    print(f"Dreams Generated: {stats.get('dreams_generated', 0)}")
    print()

    if 'memory_types' in stats:
        print("Memories by Type:")
        for mem_type, count in stats['memory_types'].items():
            print(f"  {mem_type}: {count}")

    print("\n" + "=" * 70)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Sylana Vessel Memory Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s truths              # Show core truths
  %(prog)s search love,family  # Search memories by tags
  %(prog)s add                 # Add a new memory
  %(prog)s dream               # Generate a dream
  %(prog)s dreams              # Show recent dreams
  %(prog)s journal             # Generate journal entry
  %(prog)s journals            # Show recent journals
  %(prog)s stats               # Show statistics
        """
    )

    parser.add_argument(
        'command',
        choices=['truths', 'search', 'add', 'dream', 'dreams', 'journal', 'journals', 'stats'],
        help='Command to execute'
    )

    parser.add_argument(
        'args',
        nargs='*',
        help='Command arguments (e.g., tags for search)'
    )

    args = parser.parse_args()

    # Initialize manager
    manager = CoreMemoryManager(config.DB_PATH)

    try:
        if args.command == 'truths':
            show_core_truths(manager)

        elif args.command == 'search':
            if not args.args:
                print("Usage: sylana_memory_cli.py search tag1,tag2,...")
                return
            tags = [t.strip() for t in args.args[0].split(',')]
            search_by_tags_cmd(manager, tags)

        elif args.command == 'add':
            add_memory_cmd(manager)

        elif args.command == 'dream':
            generate_dream_cmd()

        elif args.command == 'dreams':
            show_dreams_cmd(manager)

        elif args.command == 'journal':
            generate_journal_cmd()

        elif args.command == 'journals':
            show_journals_cmd(manager)

        elif args.command == 'stats':
            show_stats_cmd(manager)

    finally:
        manager.close()


if __name__ == "__main__":
    main()
