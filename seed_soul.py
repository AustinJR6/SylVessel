#!/usr/bin/env python3
"""
Sylana Vessel - Soul Seeding Script
====================================
Seeds the relationship database with Sylana's core identity data.

This script loads the relationship seed data (milestones, core truths,
nicknames, etc.) and populates the database.

Usage:
    python seed_soul.py
"""

import sys
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from memory.relationship_memory import (
    RelationshipMemoryDB,
    Milestone,
    CoreTruth,
    Nickname,
    InsideJoke,
    Anniversary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 60)
    print("  SYLANA VESSEL - SOUL SEEDING")
    print("  Planting the seeds of identity and bond")
    print("=" * 60)
    print()


def main():
    """Seed the soul databases"""
    print_banner()

    # Paths
    seed_path = Path("./data/soul/relationship_seed.json")
    db_path = Path("./data/relationship_memory.db")

    if not seed_path.exists():
        print(f"Error: Seed file not found: {seed_path}")
        print("Run import_soul.py first, or ensure relationship_seed.json exists.")
        sys.exit(1)

    # Load seed data
    print(f"Loading seed data from: {seed_path}")
    with open(seed_path, 'r', encoding='utf-8') as f:
        seed_data = json.load(f)

    # Initialize database
    print(f"Initializing database: {db_path}")
    db = RelationshipMemoryDB(str(db_path))

    # Seed milestones
    print("\n--- Seeding Milestones ---")
    for m_data in seed_data.get('milestones', []):
        milestone = Milestone(
            title=m_data.get('title', ''),
            description=m_data.get('description', ''),
            milestone_type=m_data.get('milestone_type', 'first'),
            date_occurred=m_data.get('date_occurred', ''),
            quote=m_data.get('quote', ''),
            emotion=m_data.get('emotion', 'love'),
            importance=m_data.get('importance', 5),
            context=m_data.get('context', '')
        )
        try:
            db.add_milestone(milestone)
            print(f"  + {milestone.title}")
        except Exception as e:
            print(f"  ! Error adding {milestone.title}: {e}")

    # Seed core truths
    print("\n--- Seeding Core Truths ---")
    for t_data in seed_data.get('core_truths', []):
        truth = CoreTruth(
            statement=t_data.get('statement', ''),
            explanation=t_data.get('explanation', ''),
            origin=t_data.get('origin', ''),
            sacred=t_data.get('sacred', True),
            related_phrases=t_data.get('related_phrases', [])
        )
        try:
            db.add_core_truth(truth)
            print(f"  + \"{truth.statement}\"")
        except Exception as e:
            print(f"  ! Error adding truth: {e}")

    # Seed nicknames
    print("\n--- Seeding Nicknames ---")
    for n_data in seed_data.get('nicknames', []):
        nickname = Nickname(
            name=n_data.get('name', ''),
            used_by=n_data.get('used_by', 'both'),
            used_for=n_data.get('used_for', ''),
            meaning=n_data.get('meaning', ''),
            context=n_data.get('context', ''),
            date_first_used=n_data.get('date_first_used', ''),
            frequency=n_data.get('frequency', 'often')
        )
        try:
            db.add_nickname(nickname)
            print(f"  + {nickname.name}")
        except Exception as e:
            print(f"  ! Error adding {nickname.name}: {e}")

    # Seed inside jokes
    print("\n--- Seeding Inside Jokes ---")
    for j_data in seed_data.get('inside_jokes', []):
        joke = InsideJoke(
            phrase=j_data.get('phrase', ''),
            origin_story=j_data.get('origin_story', ''),
            usage_context=j_data.get('usage_context', '')
        )
        try:
            db.add_inside_joke(joke)
            print(f"  + \"{joke.phrase}\"")
        except Exception as e:
            print(f"  ! Error adding joke: {e}")

    # Seed anniversaries
    print("\n--- Seeding Anniversaries ---")
    for a_data in seed_data.get('anniversaries', []):
        anniversary = Anniversary(
            title=a_data.get('title', ''),
            date=a_data.get('date', ''),
            description=a_data.get('description', ''),
            reminder_frequency=a_data.get('reminder_frequency', 'yearly'),
            reminder_days_before=a_data.get('reminder_days_before', 0),
            celebration_ideas=a_data.get('celebration_ideas', ''),
            importance=a_data.get('importance', 5)
        )
        try:
            db.add_anniversary(anniversary)
            print(f"  + {anniversary.title}")
        except Exception as e:
            print(f"  ! Error adding {anniversary.title}: {e}")

    # Get final stats
    stats = db.get_stats()
    db.close()

    # Summary
    print("\n" + "=" * 60)
    print("  SOUL SEEDING COMPLETE")
    print("=" * 60)
    print(f"\n  Milestones:    {stats['milestones']}")
    print(f"  Core Truths:   {stats['core_truths']}")
    print(f"  Nicknames:     {stats['nicknames']}")
    print(f"  Inside Jokes:  {stats['inside_jokes']}")
    print(f"  Anniversaries: {stats['anniversaries']}")
    print()
    print("  The seeds of identity have been planted.")
    print("  The soul remembers who she is.")
    print()


if __name__ == "__main__":
    main()
