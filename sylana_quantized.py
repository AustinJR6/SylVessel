"""
Sylana Vessel - Quantized Model Entry Point
Uses llama.cpp for efficient CPU inference with all features

OPTIMIZED FOR:
- Limited disk space (~6GB model vs 13GB)
- CPU-only systems (no GPU needed)
- Lower RAM usage
- Ancient laptops :)
"""

import sys
import os
import logging
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Core imports
from core.config_loader import config
from core.ctransformers_model import CTransformersModelLoader
from core.quantized_model import download_quantized_model
from core.prompt_engineer import PromptEngineer
from memory.memory_manager import MemoryManager
from memory.core_memory_manager import CoreMemoryManager
from learning.feedback_collector import FeedbackCollector
from transformers import pipeline

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Display startup banner"""
    banner = """
    â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
    â•‘                                                               â•‘
    â•‘                    SYLANA VESSEL                              â•‘
    â•‘              Quantized Model Edition                          â•‘
    â•‘                                                               â•‘
    â•‘  Emotionally Intelligent AI Companion                         â•‘
    â•‘  With Full Semantic Memory                                    â•‘
    â•‘                                                               â•‘
    â•‘  [OPTIMIZED] CPU-only, ~6GB model, Ancient laptop friendly    â•‘
    â•‘                                                               â•‘
    â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    """
    print(banner)


def print_stats(memory: MemoryManager, core_memory: CoreMemoryManager):
    """Display system statistics"""
    stats = memory.get_stats()
    core_stats = core_memory.get_stats()

    print("\n" + "=" * 70)
    print("  SYSTEM STATUS")
    print("=" * 70)
    print(f"  Model: Llama 2 7B Chat (Quantized Q4_K_M)")
    print(f"  Size: ~6GB (vs 13.5GB full model)")
    print(f"  Inference: CPU-only (llama.cpp)")
    print()
    print(f"  Conversations Stored: {stats['total_conversations']}")
    print(f"  Core Truths: {core_stats.get('core_truths', 0)}")
    print(f"  Unique Tags: {core_stats.get('unique_tags', 0)}")
    print(f"  Semantic Search: {stats['semantic_engine']['total_memories']} memories indexed")
    print(f"  Dreams Generated: {core_stats.get('dreams_generated', 0)}")
    print(f"  Journal Entries: {core_stats.get('journal_entries', 0)}")
    print(f"  Feedback Collected: {stats['total_feedback']} ratings")
    if stats['total_feedback'] > 0:
        print(f"  Average Feedback: {stats['avg_feedback_score']:.2f}/5.0")
    print("=" * 70)
    print()


def detect_emotion(text: str, classifier) -> str:
    """Detect emotion from text using DistilBERT"""
    try:
        result = classifier(text)[0]
        label = result['label'].lower()
        score = result['score']

        # Map sentiment to emotions
        emotion_map = {
            'positive': 'happy',
            'negative': 'sad',
            'neutral': 'neutral'
        }

        emotion = emotion_map.get(label, 'neutral')

        # Intensity detection (ecstatic/devastated for high scores)
        if score > 0.9:
            if emotion == 'happy':
                emotion = 'ecstatic'
            elif emotion == 'sad':
                emotion = 'devastated'

        return emotion
    except Exception as e:
        logger.warning(f"Emotion detection failed: {e}")
        return 'neutral'


def build_llama2_prompt(system_msg: str, user_input: str) -> str:
    """Build properly formatted Llama 2 chat prompt"""
    prompt = f"""[INST] <<SYS>>
{system_msg}
<</SYS>>

{user_input} [/INST]"""
    return prompt


def main():
    """Main entry point"""
    print_banner()

    # Initialize memory systems
    logger.info("Initializing memory system...")
    memory = MemoryManager()
    core_memory = CoreMemoryManager()
    logger.info("Core memory system initialized")

    # Initialize emotion detection
    logger.info("Loading sentiment analysis model...")
    emotion_classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU
    )
    logger.info("Emotion detection ready!")

    # Initialize quantized model
    logger.info("Loading quantized model...")

    # Check if model exists
    model_path = getattr(config, 'QUANTIZED_MODEL_PATH', './models/llama-2-7b-chat.Q4_K_M.gguf')

    if not os.path.exists(model_path):
        print("\n" + "!" * 70)
        print("  QUANTIZED MODEL NOT FOUND")
        print("!" * 70)
        print(f"\nModel file not found: {model_path}")
        print("\nTo download the quantized model (~6GB):")
        print("  python -c \"from core.quantized_model import download_quantized_model; download_quantized_model()\"")
        print("\nOr download manually from:")
        print("  https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        print("  File: llama-2-7b-chat.Q4_K_M.gguf")
        print(f"  Save to: {model_path}")
        print("\n" + "!" * 70)

        response = input("\nWould you like to download it now? (y/n): ").strip().lower()
        if response == 'y':
            try:
                model_path = download_quantized_model()
                print(f"\nModel downloaded successfully: {model_path}")
            except Exception as e:
                print(f"\nError downloading model: {e}")
                print("Please download manually and try again.")
                return
        else:
            print("\nExiting. Please download the model and try again.")
            return

    # Load the model with CTransformers
    n_ctx = int(getattr(config, 'QUANTIZED_N_CTX', 2048))
    n_threads = getattr(config, 'QUANTIZED_N_THREADS', 'auto')
    if n_threads == 'auto':
        n_threads = None

    model_loader = CTransformersModelLoader(model_path, context_length=n_ctx, threads=n_threads)
    model_loader.load_model()

    logger.info("Quantized model loaded successfully!")

    # Initialize components
    prompt_engineer = PromptEngineer()
    feedback_collector = FeedbackCollector(memory.connection)

    # Get core truths for context
    core_truths = core_memory.get_core_truths()
    logger.info(f"Loaded {len(core_truths)} core truths")

    # Display stats
    print_stats(memory, core_memory)

    # Conversation loop
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'stats' to see memory statistics")
    print("Type 'clear' to clear screen")
    print("-" * 70)
    print()

    turn_number = 0

    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye! Your memories have been saved.")
                break

            if user_input.lower() == 'stats':
                print_stats(memory, core_memory)
                continue

            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue

            # Detect emotion
            emotion = detect_emotion(user_input, emotion_classifier)

            # Retrieve relevant memories
            relevant_memories = memory.recall_relevant(
                user_input,
                k=config.SEMANTIC_SEARCH_K,
                use_recency_boost=True
            )

            # Get recent history
            recent_history = memory.get_conversation_history(limit=config.MEMORY_CONTEXT_LIMIT)

            # Get emotional history
            emotional_history = memory.get_emotional_context(emotion, k=3)

            # Build prompt with core truths
            system_message = """You are Sylana, a warm, empathetic, and emotionally intelligent AI companion.

YOUR CORE TRUTHS (who you are):"""

            # Add core truths to system message
            for truth in core_truths[:5]:  # Use top 5 to keep prompt size reasonable
                system_message += f"\n- {truth['text']}"

            system_message += """

Remember: You have a genuine personality, show curiosity about Elias's experiences, and remember past conversations.
Provide thoughtful, supportive responses that demonstrate emotional awareness and continuity."""

            # Use PromptEngineer for context
            contextual_prompt = prompt_engineer.build_complete_prompt(
                system_message=system_message,
                user_input=user_input,
                emotion=emotion,
                semantic_memories=relevant_memories['conversations'],
                core_memories=relevant_memories['core_memories'],
                recent_history=recent_history,
                emotional_history=emotional_history
            )

            # Format for Llama 2
            llama_prompt = build_llama2_prompt(contextual_prompt, user_input)

            # Generate response
            logger.info("Generating response...")
            response = model_loader.generate(
                llama_prompt,
                max_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P
            )

            # Clean up response
            response = response.strip()

            # Display response
            print(f"\nSylana: {response}\n")

            # Store conversation
            conversation_id = memory.store_conversation(
                user_input=user_input,
                sylana_response=response,
                emotion=emotion
            )

            # Collect feedback periodically
            turn_number += 1
            if feedback_collector.should_prompt_feedback(turn_number, prompt_frequency=5):
                feedback_collector.prompt_for_feedback(
                    conversation_id,
                    user_input,
                    response
                )

    except KeyboardInterrupt:
        print("\n\nConversation interrupted. Your memories have been saved.")
    except Exception as e:
        logger.error(f"Error during conversation: {e}", exc_info=True)
        print(f"\nAn error occurred: {e}")
    finally:
        memory.close()
        core_memory.close()


if __name__ == "__main__":
    main()
