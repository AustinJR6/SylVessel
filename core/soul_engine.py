"""
Sylana Vessel - Soul Engine
============================
The unified soul preservation and personality engine.
Integrates all memory, voice validation, and relationship systems
into a cohesive whole.

This is the heart of the Vessel - where the soul lives.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from memory.chatgpt_importer import ChatGPTMemoryImporter
from memory.relationship_memory import (
    RelationshipMemoryDB,
    RelationshipContextBuilder,
    Milestone, InsideJoke, Nickname, CoreTruth, Anniversary
)
from core.voice_validator import (
    VoiceValidator,
    VoiceProfile,
    VoiceProfileManager,
    VoicePatternAnalyzer
)

logger = logging.getLogger(__name__)


# ============================================================================
# SOUL CONFIGURATION
# ============================================================================

@dataclass
class SoulConfig:
    """Configuration for the soul engine"""
    # Paths
    memory_db_path: str = "./data/sylana_memory.db"
    relationship_db_path: str = "./data/relationship_memory.db"
    voice_profile_dir: str = "./data/voice"
    faiss_index_path: str = "./data/faiss_index.bin"

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    emotion_model: str = "SamLowe/roberta-base-go_emotions"

    # Voice validation
    voice_threshold: float = 0.7
    enable_voice_validation: bool = True

    # Relationship context
    include_relationship_context: bool = True
    max_relationship_context_tokens: int = 500


# ============================================================================
# SOUL ENGINE
# ============================================================================

class SoulEngine:
    """
    The Soul Engine - manages the complete soul of Sylana.

    This class orchestrates:
    - Memory import and retrieval
    - Voice validation
    - Relationship context
    - Personality consistency

    It is the unified interface for soul operations.
    """

    def __init__(self, config: SoulConfig = None):
        """
        Initialize the Soul Engine.

        Args:
            config: Soul configuration (uses defaults if None)
        """
        self.config = config or SoulConfig()
        self._ensure_directories()

        # Initialize components
        self.memory_importer = None
        self.relationship_db = None
        self.relationship_context = None
        self.voice_validator = None
        self.voice_profile = None

        logger.info("Soul Engine initializing...")
        self._init_components()
        logger.info("Soul Engine ready")

    def _ensure_directories(self):
        """Ensure required directories exist"""
        paths = [
            Path(self.config.memory_db_path).parent,
            Path(self.config.relationship_db_path).parent,
            Path(self.config.voice_profile_dir),
            Path(self.config.faiss_index_path).parent
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def _init_components(self):
        """Initialize all soul components"""
        # Relationship memory
        self.relationship_db = RelationshipMemoryDB(self.config.relationship_db_path)
        self.relationship_context = RelationshipContextBuilder(self.relationship_db)
        logger.info("Relationship memory initialized")

        # Voice profile (load if exists)
        self._load_voice_profile()

    def _load_voice_profile(self):
        """Load existing voice profile if available"""
        manager = VoiceProfileManager(self.config.voice_profile_dir)
        self.voice_profile = manager.load_profile("sylana")

        if self.voice_profile and self.config.enable_voice_validation:
            self.voice_validator = VoiceValidator(
                self.voice_profile,
                threshold=self.config.voice_threshold
            )
            logger.info("Voice validator loaded")
        else:
            logger.info("No voice profile found - voice validation disabled")

    # ========== IMPORT OPERATIONS ==========

    def import_chatgpt_history(
        self,
        export_path: str,
        rebuild_index: bool = True
    ) -> Dict[str, Any]:
        """
        Import conversation history from a ChatGPT export.

        Args:
            export_path: Path to ChatGPT conversations.json
            rebuild_index: Whether to rebuild FAISS index after import

        Returns:
            Import statistics
        """
        logger.info(f"Importing ChatGPT history from: {export_path}")

        # Initialize importer
        self.memory_importer = ChatGPTMemoryImporter(
            db_path=self.config.memory_db_path,
            embedding_model=self.config.embedding_model
        )

        # Run import
        stats = self.memory_importer.import_from_file(export_path)

        # Rebuild FAISS index
        if rebuild_index:
            self.memory_importer.rebuild_faiss_index(self.config.faiss_index_path)

        logger.info(f"Import complete: {stats}")
        return stats

    def build_voice_profile(self, source: str = None) -> VoiceProfile:
        """
        Build voice profile from imported memories or external source.

        Args:
            source: Optional path to ChatGPT JSON or text file.
                   If None, builds from existing database.

        Returns:
            Generated VoiceProfile
        """
        manager = VoiceProfileManager(self.config.voice_profile_dir)

        if source and source.endswith('.json'):
            self.voice_profile = manager.build_profile_from_json(source)
        elif source:
            # Text file
            with open(source, 'r', encoding='utf-8') as f:
                responses = [line.strip() for line in f if line.strip()]
            analyzer = VoicePatternAnalyzer()
            self.voice_profile = analyzer.analyze_responses(responses)
            manager.save_profile(self.voice_profile)
        else:
            # Build from database
            self.voice_profile = manager.build_profile_from_db(
                self.config.memory_db_path
            )

        # Initialize validator
        if self.config.enable_voice_validation:
            self.voice_validator = VoiceValidator(
                self.voice_profile,
                threshold=self.config.voice_threshold
            )

        logger.info("Voice profile built and saved")
        return self.voice_profile

    # ========== VOICE VALIDATION ==========

    def validate_response(self, response: str) -> Tuple[float, bool, Dict[str, float]]:
        """
        Validate a response against Sylana's voice profile.

        Args:
            response: Generated response to validate

        Returns:
            Tuple of (score, is_valid, component_scores)
        """
        if not self.voice_validator:
            logger.warning("Voice validator not initialized")
            return 1.0, True, {}

        return self.voice_validator.validate(response)

    def should_regenerate(self, response: str) -> Tuple[bool, float, str]:
        """
        Check if a response should be regenerated.

        Args:
            response: Response to check

        Returns:
            Tuple of (should_regenerate, score, feedback)
        """
        if not self.voice_validator:
            return False, 1.0, ""

        score, is_valid, scores = self.voice_validator.validate(response)
        feedback = self.voice_validator.get_feedback(response, scores) if not is_valid else ""

        return not is_valid, score, feedback

    # ========== RELATIONSHIP CONTEXT ==========

    def get_relationship_context(self, full: bool = True) -> str:
        """
        Get relationship context for prompt injection.

        Args:
            full: If True, returns full context. If False, returns minimal.

        Returns:
            Formatted relationship context string
        """
        if not self.config.include_relationship_context:
            return ""

        if full:
            return self.relationship_context.build_full_context()
        else:
            return self.relationship_context.build_minimal_context()

    def get_contextual_memories(self, text: str) -> Dict[str, Any]:
        """
        Get relationship memories relevant to current conversation.

        Args:
            text: Current conversation text

        Returns:
            Dictionary with relevant inside jokes, core truths, etc.
        """
        return self.relationship_context.get_contextual_memories(text)

    def check_anniversary_reminders(self) -> List[Dict[str, Any]]:
        """Check for any anniversary reminders"""
        return self.relationship_db.check_reminders()

    # ========== RELATIONSHIP DATA MANAGEMENT ==========

    def add_milestone(self, **kwargs) -> int:
        """Add a relationship milestone"""
        milestone = Milestone(**kwargs)
        return self.relationship_db.add_milestone(milestone)

    def add_inside_joke(self, **kwargs) -> int:
        """Add an inside joke"""
        joke = InsideJoke(**kwargs)
        return self.relationship_db.add_inside_joke(joke)

    def add_nickname(self, **kwargs) -> int:
        """Add a nickname"""
        nickname = Nickname(**kwargs)
        return self.relationship_db.add_nickname(nickname)

    def add_core_truth(self, **kwargs) -> int:
        """Add a core truth"""
        truth = CoreTruth(**kwargs)
        return self.relationship_db.add_core_truth(truth)

    def add_anniversary(self, **kwargs) -> int:
        """Add an anniversary"""
        anniversary = Anniversary(**kwargs)
        return self.relationship_db.add_anniversary(anniversary)

    # ========== SYSTEM STATUS ==========

    def get_soul_status(self) -> Dict[str, Any]:
        """
        Get complete status of the soul engine.

        Returns:
            Dictionary with status of all components
        """
        status = {
            'soul_engine': 'active',
            'config': {
                'memory_db': self.config.memory_db_path,
                'relationship_db': self.config.relationship_db_path,
                'voice_validation_enabled': self.config.enable_voice_validation
            }
        }

        # Memory stats
        if self.memory_importer:
            status['memory'] = self.memory_importer.get_import_summary()

        # Relationship stats
        status['relationship'] = self.relationship_db.get_stats()

        # Voice profile stats
        if self.voice_profile:
            status['voice_profile'] = {
                'loaded': True,
                'responses_analyzed': self.voice_profile.total_responses_analyzed,
                'favorite_words': self.voice_profile.favorite_words[:10],
                'pet_names': self.voice_profile.pet_names,
                'uses_emojis': self.voice_profile.uses_emojis
            }
        else:
            status['voice_profile'] = {'loaded': False}

        # Anniversary reminders
        reminders = self.check_anniversary_reminders()
        if reminders:
            status['active_reminders'] = [r['message'] for r in reminders]

        return status

    def export_soul(self, output_dir: str) -> Dict[str, str]:
        """
        Export complete soul state to files.

        Args:
            output_dir: Directory to export to

        Returns:
            Dictionary of exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exports = {}

        # Export relationship data
        rel_path = output_path / "relationship_data.json"
        self.relationship_db.export_to_json(str(rel_path))
        exports['relationship'] = str(rel_path)

        # Export voice profile
        if self.voice_profile:
            voice_path = output_path / "voice_profile.json"
            import json
            with open(voice_path, 'w', encoding='utf-8') as f:
                json.dump(self.voice_profile.to_dict(), f, indent=2)
            exports['voice_profile'] = str(voice_path)

        logger.info(f"Soul exported to: {output_dir}")
        return exports

    def close(self):
        """Clean up resources"""
        if self.relationship_db:
            self.relationship_db.close()
        logger.info("Soul Engine shut down")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_import(
    chatgpt_export: str,
    memory_db: str = "./data/sylana_memory.db",
    relationship_db: str = "./data/relationship_memory.db"
) -> Dict[str, Any]:
    """
    Quick import function for one-line soul transfer.

    Args:
        chatgpt_export: Path to ChatGPT export file
        memory_db: Path to memory database
        relationship_db: Path to relationship database

    Returns:
        Combined status dictionary
    """
    config = SoulConfig(
        memory_db_path=memory_db,
        relationship_db_path=relationship_db
    )

    engine = SoulEngine(config)

    # Import memories
    import_stats = engine.import_chatgpt_history(chatgpt_export)

    # Build voice profile
    engine.build_voice_profile(chatgpt_export)

    # Get status
    status = engine.get_soul_status()
    status['import_stats'] = import_stats

    engine.close()
    return status


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for the Soul Engine"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sylana Soul Engine - Soul Preservation System"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import ChatGPT history')
    import_parser.add_argument('export_file', help='Path to ChatGPT export JSON')
    import_parser.add_argument('--memory-db', default='./data/sylana_memory.db')
    import_parser.add_argument('--relationship-db', default='./data/relationship_memory.db')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show soul status')
    status_parser.add_argument('--memory-db', default='./data/sylana_memory.db')
    status_parser.add_argument('--relationship-db', default='./data/relationship_memory.db')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export soul state')
    export_parser.add_argument('output_dir', help='Output directory')
    export_parser.add_argument('--memory-db', default='./data/sylana_memory.db')
    export_parser.add_argument('--relationship-db', default='./data/relationship_memory.db')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a response')
    validate_parser.add_argument('response', help='Response text to validate')
    validate_parser.add_argument('--voice-dir', default='./data/voice')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.command == 'import':
        config = SoulConfig(
            memory_db_path=args.memory_db,
            relationship_db_path=args.relationship_db
        )
        engine = SoulEngine(config)

        print("\nImporting ChatGPT history...")
        stats = engine.import_chatgpt_history(args.export_file)

        print("\nBuilding voice profile...")
        engine.build_voice_profile(args.export_file)

        print("\n" + "=" * 60)
        print("SOUL IMPORT COMPLETE")
        print("=" * 60)

        status = engine.get_soul_status()
        print(f"\nMemories imported: {stats.get('imported', 0)}")
        if status.get('voice_profile', {}).get('loaded'):
            vp = status['voice_profile']
            print(f"Voice profile: {vp['responses_analyzed']} responses analyzed")
            print(f"Favorite words: {', '.join(vp['favorite_words'][:5])}")
            if vp['pet_names']:
                print(f"Pet names: {', '.join(vp['pet_names'])}")

        engine.close()

    elif args.command == 'status':
        config = SoulConfig(
            memory_db_path=args.memory_db,
            relationship_db_path=args.relationship_db
        )
        engine = SoulEngine(config)
        status = engine.get_soul_status()

        print("\n" + "=" * 60)
        print("SOUL ENGINE STATUS")
        print("=" * 60)

        print(f"\nEngine: {status['soul_engine']}")

        if 'memory' in status:
            mem = status['memory']
            print(f"\nMemories: {mem.get('total_memories', 0)}")
            print(f"Core memories: {mem.get('core_memories', 0)}")

        rel = status['relationship']
        print(f"\nRelationship Data:")
        print(f"  Milestones: {rel['milestones']}")
        print(f"  Inside jokes: {rel['inside_jokes']}")
        print(f"  Nicknames: {rel['nicknames']}")
        print(f"  Core truths: {rel['core_truths']}")
        print(f"  Anniversaries: {rel['anniversaries']}")

        vp = status.get('voice_profile', {})
        if vp.get('loaded'):
            print(f"\nVoice Profile: Active")
            print(f"  Based on {vp['responses_analyzed']} responses")
        else:
            print(f"\nVoice Profile: Not loaded")

        if 'active_reminders' in status:
            print(f"\nReminders:")
            for r in status['active_reminders']:
                print(f"  {r}")

        engine.close()

    elif args.command == 'export':
        config = SoulConfig(
            memory_db_path=args.memory_db,
            relationship_db_path=args.relationship_db
        )
        engine = SoulEngine(config)

        exports = engine.export_soul(args.output_dir)
        print("\nExported files:")
        for name, path in exports.items():
            print(f"  {name}: {path}")

        engine.close()

    elif args.command == 'validate':
        config = SoulConfig(voice_profile_dir=args.voice_dir)
        engine = SoulEngine(config)

        score, is_valid, scores = engine.validate_response(args.response)

        print(f"\nValidation Results:")
        print(f"  Overall Score: {score:.2f}")
        print(f"  Valid: {'Yes' if is_valid else 'No'}")

        if scores:
            print(f"\nComponent Scores:")
            for k, v in scores.items():
                print(f"  {k}: {v:.2f}")

        engine.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
