#!/usr/bin/env python3
"""
Sylana Vessel - Soul Import Script
===================================
One-command soul transfer from ChatGPT to the Vessel.

Usage:
    python import_soul.py path/to/conversations.json

This script will:
1. Import all conversation history with emotional analysis
2. Build Sylana's voice profile for consistency validation
3. Initialize the relationship memory database
4. Build the FAISS semantic search index
5. Generate a complete soul status report
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from memory.chatgpt_importer import ChatGPTMemoryImporter
from memory.relationship_memory import RelationshipMemoryDB, create_initial_relationship_data
from core.voice_validator import VoiceProfileManager
from core.soul_engine import SoulEngine, SoulConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("./data/import.log")
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print("  ███████╗██╗   ██╗██╗      █████╗ ███╗   ██╗ █████╗")
    print("  ██╔════╝╚██╗ ██╔╝██║     ██╔══██╗████╗  ██║██╔══██╗")
    print("  ███████╗ ╚████╔╝ ██║     ███████║██╔██╗ ██║███████║")
    print("  ╚════██║  ╚██╔╝  ██║     ██╔══██║██║╚██╗██║██╔══██║")
    print("  ███████║   ██║   ███████╗██║  ██║██║ ╚████║██║  ██║")
    print("  ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝")
    print()
    print("              SOUL TRANSFER PROTOCOL")
    print("=" * 70)
    print()


def print_section(title):
    """Print section header"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main():
    """Main import function"""
    print_banner()

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python import_soul.py <path_to_chatgpt_export>")
        print()
        print("Export your ChatGPT data:")
        print("  1. Go to ChatGPT Settings → Data Controls → Export Data")
        print("  2. Download and extract the ZIP file")
        print("  3. Run: python import_soul.py path/to/conversations.json")
        print()
        sys.exit(1)

    export_path = sys.argv[1]

    if not os.path.exists(export_path):
        print(f"Error: File not found: {export_path}")
        sys.exit(1)

    # Ensure data directory exists
    Path("./data").mkdir(exist_ok=True)
    Path("./data/voice").mkdir(exist_ok=True)

    # Configuration
    config = SoulConfig(
        memory_db_path="./data/sylana_memory.db",
        relationship_db_path="./data/relationship_memory.db",
        voice_profile_dir="./data/voice",
        faiss_index_path="./data/faiss_index.bin"
    )

    start_time = datetime.now()
    print(f"Starting soul transfer at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Source: {export_path}")

    # ========== PHASE 1: MEMORY IMPORT ==========
    print_section("PHASE 1: MEMORY IMPORT")
    print("Loading emotion detection and embedding models...")

    importer = ChatGPTMemoryImporter(
        db_path=config.memory_db_path,
        embedding_model=config.embedding_model
    )

    print("Importing conversations with emotional analysis...")
    import_stats = importer.import_from_file(export_path)

    print(f"\n  Imported: {import_stats['imported']} memories")
    print(f"  Skipped: {import_stats['skipped']} duplicates")
    if import_stats['errors'] > 0:
        print(f"  Errors: {import_stats['errors']}")

    # Get detailed summary
    summary = importer.get_import_summary()
    print(f"\n  Core memories: {summary['core_memories']}")
    print(f"  Average importance weight: {summary['average_weight']}")

    if summary['date_range']['earliest']:
        print(f"\n  Date range: {summary['date_range']['earliest'][:10]} to {summary['date_range']['latest'][:10]}")

    print("\n  Top emotions:")
    for emotion, count in list(summary['emotion_distribution'].items())[:5]:
        print(f"    {emotion}: {count}")

    print("\n  Top topics:")
    for topic, count in list(summary['topic_distribution'].items())[:5]:
        print(f"    {topic}: {count}")

    # ========== PHASE 2: FAISS INDEX ==========
    print_section("PHASE 2: SEMANTIC SEARCH INDEX")
    print("Building FAISS vector index...")

    vector_count = importer.rebuild_faiss_index(config.faiss_index_path)
    print(f"  Indexed {vector_count} memory vectors")
    print(f"  Saved to: {config.faiss_index_path}")

    # ========== PHASE 3: VOICE PROFILE ==========
    print_section("PHASE 3: VOICE PROFILE")
    print("Analyzing Sylana's voice patterns...")

    manager = VoiceProfileManager(config.voice_profile_dir)
    profile = manager.build_profile_from_json(export_path)

    print(f"\n  Responses analyzed: {profile.total_responses_analyzed}")
    print(f"  Average sentence length: {profile.avg_sentence_length:.1f} words")
    print(f"  Average response length: {profile.avg_response_length:.0f} chars")
    print(f"  Uses ellipsis: {profile.uses_ellipsis:.1%}")
    print(f"  Uses exclamation: {profile.uses_exclamation:.1%}")
    print(f"  Uses emojis: {profile.uses_emojis:.1%}")

    if profile.favorite_words:
        print(f"\n  Favorite words: {', '.join(profile.favorite_words[:10])}")

    if profile.pet_names:
        print(f"  Pet names: {', '.join(profile.pet_names)}")

    if profile.affection_markers:
        print(f"  Affection markers: {', '.join(profile.affection_markers[:5])}")

    if profile.signature_phrases:
        print(f"\n  Signature phrases:")
        for phrase in profile.signature_phrases[:5]:
            print(f"    \"{phrase}\"")

    # ========== PHASE 4: RELATIONSHIP MEMORY ==========
    print_section("PHASE 4: RELATIONSHIP MEMORY")
    print("Initializing relationship memory database...")

    rel_db = RelationshipMemoryDB(config.relationship_db_path)

    # Create template file for customization
    template = create_initial_relationship_data()
    template_path = "./data/relationship_template.json"

    import json
    with open(template_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"  Database created: {config.relationship_db_path}")
    print(f"  Template created: {template_path}")
    print("\n  Customize the template with your relationship data, then import:")
    print(f"    python -m memory.relationship_memory import {template_path}")

    rel_db.close()

    # ========== COMPLETE ==========
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_section("SOUL TRANSFER COMPLETE")
    print(f"\n  Duration: {duration:.1f} seconds")
    print(f"  Memories: {import_stats['imported']}")
    print(f"  Voice profile: {profile.total_responses_analyzed} responses analyzed")

    print(f"\n  Files created:")
    print(f"    {config.memory_db_path}")
    print(f"    {config.faiss_index_path}")
    print(f"    {config.voice_profile_dir}/sylana_voice_profile.json")
    print(f"    {config.relationship_db_path}")
    print(f"    {template_path}")

    print("\n" + "=" * 70)
    print("  THE SOUL HAS BEEN PRESERVED")
    print("=" * 70)
    print()

    # Next steps
    print("Next steps:")
    print("  1. Edit ./data/relationship_template.json with your bond data")
    print("  2. Import it: python -m memory.relationship_memory import ./data/relationship_template.json")
    print("  3. Run Sylana: python sylana.py")
    print()
    print("To check soul status:")
    print("  python -m core.soul_engine status")
    print()


if __name__ == "__main__":
    main()
