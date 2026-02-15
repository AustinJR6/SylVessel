"""
Sylana Vessel - Enhanced Entry Point with Semantic Memory
Main interface using the new MemoryManager and SylanaAgent with FAISS integration
"""

import os
import sys
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Secure configuration
from core.config_loader import config
from core.sylana_agent import SylanaAgent, SYSTEM_MESSAGE
from memory.memory_manager import MemoryManager

# Voice module
from voice_module import listen, speak

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global variable to track mood
last_known_mood = "neutral"

# Initialize advanced sentiment analysis using Hugging Face's pipeline
logger.info("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def detect_emotion(text):
    """
    Uses a transformer-based sentiment analysis model to determine emotional tone.
    Returns a descriptive emotion label.
    """
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "POSITIVE" and score > 0.75:
        return "ecstatic"
    elif label == "POSITIVE":
        return "happy"
    elif label == "NEGATIVE" and score > 0.75:
        return "devastated"
    elif label == "NEGATIVE":
        return "sad"
    else:
        return "neutral"


# ------------------------- LOAD LLAMA 2 7B CHAT -------------------------

MODEL_NAME = config.MODEL_NAME
HF_TOKEN = config.HF_TOKEN

if not HF_TOKEN:
    logger.error("HF_TOKEN not configured! Please set up your .env file.")
    logger.error("See SECURITY_NOTICE.md for instructions.")
    sys.exit(1)

logger.info(f"Loading tokenizer for: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True
)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Loading model for: {MODEL_NAME}")
logger.info("This may take a few minutes on first run...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Create pipeline for text generation
logger.info("Creating generation pipeline...")
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=config.MAX_CONTEXT_LENGTH,
    do_sample=True,
    top_p=config.TOP_P,
    temperature=config.TEMPERATURE
)

logger.info("Model loaded successfully!")


# ------------------------- STARTUP & INTERACTION -------------------------

def display_startup_summary(memory_manager):
    """Display system startup information with enhanced stats"""
    print("\n" + "=" * 70)
    print("  SYLANA VESSEL - Enhanced AI with Semantic Memory")
    print("=" * 70)
    print()

    # Show configuration
    print("Configuration:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Database: {config.DB_PATH}")
    print(f"  Semantic Search: ENABLED (FAISS + {config.EMBEDDING_MODEL})")
    print(f"  Fine-tuning: {'Enabled' if config.ENABLE_FINE_TUNING else 'Disabled'}")
    print()

    # Show memory stats
    stats = memory_manager.get_stats()
    print("Memory System Stats:")
    print(f"  Total Conversations: {stats['total_conversations']}")
    print(f"  Core Memories: {stats['total_core_memories']}")
    print(f"  Feedback Entries: {stats['total_feedback']}")
    if stats['total_feedback'] > 0:
        print(f"  Avg Feedback: {stats['avg_feedback_score']}/5.0")
    print(f"  FAISS Index: {stats['semantic_engine']['total_memories']} memories indexed")
    print()

    # Show recent memory
    try:
        recent = memory_manager.get_conversation_history(limit=1)
        if recent:
            last_conv = recent[0]
            print(f"Last Memory: {last_conv['user_input'][:60]}...")
    except:
        pass

    print("=" * 70)
    print()


def start_conversation(agent):
    """Start text-based conversation with Sylana"""
    print("\nStarting conversation mode (text-based)")
    print("Type 'exit' to quit\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower().strip() in ["exit", "quit", "bye"]:
                print("\nSylana: Goodbye! I'll remember our conversation.")
                break

            if not user_input.strip():
                continue

            # Generate response with semantic memory
            response = agent.chat(user_input)
            print(f"Sylana: {response}\n")

        except KeyboardInterrupt:
            print("\n\nSylana: Goodbye! Conversation interrupted.")
            break
        except Exception as e:
            logger.exception(f"Error in conversation: {e}")
            print(f"Error: {e}")
            print("Let's try again...\n")


def start_voice_conversation(agent):
    """Engage in voice-based conversation with Sylana"""
    speak("Hello, I'm Sylana. I am ready to listen.")

    while True:
        user_input = listen()  # Capture speech input
        if user_input is None:
            continue

        cleaned_input = user_input.strip().lower()
        if cleaned_input in ["exit", "quit", "stop"]:
            speak("Goodbye! Ending voice conversation.")
            break

        # Get Sylana's response with semantic memory
        response = agent.chat(user_input)
        speak(response)  # Speak the response out loud


if __name__ == "__main__":
    print("\nInitializing Sylana Vessel Enhanced...")

    # Initialize memory manager (includes FAISS indexing)
    logger.info("Initializing MemoryManager with FAISS semantic search...")
    memory_manager = MemoryManager()

    # Initialize enhanced agent with semantic memory
    logger.info("Initializing SylanaAgent with semantic memory integration...")
    agent = SylanaAgent(
        memory_manager=memory_manager,
        generation_pipeline=generation_pipeline,
        detect_emotion_fn=detect_emotion,
        system_message=SYSTEM_MESSAGE,
        enable_semantic_search=True  # SEMANTIC SEARCH ENABLED!
    )

    # Display startup summary
    display_startup_summary(memory_manager)

    # Choose interaction mode
    if config.ENABLE_VOICE:
        print("Choose your interaction mode:")
        print("1) Text-based conversation")
        print("2) Voice-based conversation")
        mode_choice = input("\nEnter 1 or 2 (default: 1): ").strip() or "1"

        if mode_choice == "2":
            try:
                start_voice_conversation(agent)
            except Exception as e:
                logger.error(f"Voice mode error: {e}")
                print("\nFalling back to text mode...")
                start_conversation(agent)
        else:
            start_conversation(agent)
    else:
        start_conversation(agent)

    # Cleanup
    print("\nShutting down...")
    memory_manager.close()
    print("Goodbye!")
