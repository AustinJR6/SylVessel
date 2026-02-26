"""
Sylana Vessel - Core Module
Core AI engine, configuration, and model management
"""

from .config_loader import config
from .brain import Brain
from .logging_config import configure_logging

# Import new soul preservation components
try:
    from .voice_validator import (
        VoiceValidator,
        VoiceProfile,
        VoiceProfileManager,
        VoicePatternAnalyzer
    )
    from .soul_engine import SoulEngine, SoulConfig
    from .personality import (
        SylanaPersonality,
        PersonalityLoader,
        PersonalityPromptGenerator,
        load_sylana_personality,
        get_system_prompt,
        get_personality_reminder
    )
except ImportError:
    # Optional components - may not be available in all environments
    pass

__all__ = [
    'config',
    'Brain',
    'configure_logging',
    'VoiceValidator',
    'VoiceProfile',
    'VoiceProfileManager',
    'VoicePatternAnalyzer',
    'SoulEngine',
    'SoulConfig',
    'SylanaPersonality',
    'PersonalityLoader',
    'PersonalityPromptGenerator',
    'load_sylana_personality',
    'get_system_prompt',
    'get_personality_reminder'
]
