"""
Sylana Vessel - Secure Configuration Loader
Loads all configuration from environment variables with secure defaults
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Determine project root (where .env should be)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables from .env file
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"[OK] Loaded configuration from {ENV_FILE}")
else:
    print(f"[WARNING] No .env file found at {ENV_FILE}")
    print(f"[WARNING] Copy .env.template to .env and configure your secrets")
    print(f"[WARNING] Using default values (may not work without HF_TOKEN)")


class Config:
    """Centralized configuration management with environment variable loading"""

    def __init__(self):
        # Security: API Tokens
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
        cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
        self.CORS_ORIGINS = [o.strip() for o in cors_origins_raw.split(",") if o.strip()] or ["*"]
        if not self.HF_TOKEN:
            print("[WARNING] HF_TOKEN not set (only needed for local HuggingFace model paths)")

        # Database Configuration
        self.DB_PATH = os.getenv(
            "SYLANA_DB_PATH",
            str(PROJECT_ROOT / "data" / "sylana_memory.db")
        )
        self.SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

        # Model Configuration
        self.MODEL_NAME = os.getenv(
            "MODEL_NAME",
            "meta-llama/Llama-2-7b-chat-hf"
        )
        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Fine-Tuning Controls
        self.ENABLE_FINE_TUNING = os.getenv("ENABLE_FINE_TUNING", "false").lower() == "true"
        self.MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "100"))
        self.CHECKPOINT_DIR = os.getenv(
            "CHECKPOINT_DIR",
            str(PROJECT_ROOT / "data" / "checkpoints")
        )

        # Generation Parameters
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.9"))
        self.TOP_P = float(os.getenv("TOP_P", "0.9"))
        # Increased default to reduce clipped replies in streaming/non-streaming chat.
        self.MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "320"))
        self.MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "512"))

        # Memory Configuration
        self.MEMORY_CONTEXT_LIMIT = int(os.getenv("MEMORY_CONTEXT_LIMIT", "5"))
        self.SEMANTIC_SEARCH_K = int(os.getenv("SEMANTIC_SEARCH_K", "5"))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

        # Voice Configuration
        self.ENABLE_VOICE = os.getenv("ENABLE_VOICE", "true").lower() == "true"
        self.TTS_RATE = int(os.getenv("TTS_RATE", "160"))
        self.TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv(
            "LOG_FILE",
            str(PROJECT_ROOT / "data" / "sylana.log")
        )

        # Ensure data directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs_to_create = [
            Path(self.DB_PATH).parent,
            Path(self.CHECKPOINT_DIR),
            Path(self.LOG_FILE).parent,
            PROJECT_ROOT / "data" / "media"
        ]

        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self):
        """Validate critical configuration values"""
        errors = []

        if not self.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required for Claude API responses")

        if self.ENABLE_FINE_TUNING and self.MIN_TRAINING_SAMPLES < 10:
            errors.append("MIN_TRAINING_SAMPLES should be at least 10 for fine-tuning")

        if self.TEMPERATURE < 0 or self.TEMPERATURE > 2:
            errors.append("TEMPERATURE should be between 0 and 2")

        if errors:
            print("[WARNING] Configuration validation warnings:")
            for error in errors:
                print(f"   - {error}")
            return False

        return True

    def __repr__(self):
        """Safe representation hiding secrets"""
        token_display = f"{self.HF_TOKEN[:10]}..." if self.HF_TOKEN else "NOT_SET"
        return f"""
Sylana Vessel Configuration:
  HF_TOKEN: {token_display}
  ANTHROPIC_API_KEY: {"SET" if self.ANTHROPIC_API_KEY else "NOT_SET"}
  CLAUDE_MODEL: {self.CLAUDE_MODEL}
  CORS_ORIGINS: {','.join(self.CORS_ORIGINS)}
  DB_PATH: {self.DB_PATH}
  MODEL_NAME: {self.MODEL_NAME}
  EMBEDDING_MODEL: {self.EMBEDDING_MODEL}
  ENABLE_FINE_TUNING: {self.ENABLE_FINE_TUNING}
  CHECKPOINT_DIR: {self.CHECKPOINT_DIR}
  TEMPERATURE: {self.TEMPERATURE}
  ENABLE_VOICE: {self.ENABLE_VOICE}
  LOG_LEVEL: {self.LOG_LEVEL}
        """.strip()


# Global configuration instance
config = Config()

# Validate on import
config.validate()


if __name__ == "__main__":
    print(config)
    print("\n✅ Configuration loaded successfully")
