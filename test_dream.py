"""Quick test of dream generation"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.config_loader import config
from memory.dream.dream_generator import DreamGenerator

# Generate a dream
generator = DreamGenerator(config.DB_PATH)
dream = generator.generate_dream(['love', 'identity', 'children'])

if dream:
    print(generator.format_dream(dream))
    print("\n[OK] Dream generation successful!")
else:
    print("[ERROR] No dream generated")

generator.close()
