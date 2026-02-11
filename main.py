# main.py
"""
Sylana Vessel - Main Entry Point
Clean interface to start Sylana with various modes
"""
import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import core configuration and components
from core.config_loader import config

# Try new MemoryManager first, fall back to old system
try:
    from memory.memory_manager import MemoryManager
    use_new_memory = True
except ImportError:
    from long_term_memory import build_index, recall_memory
    from adaptive_learning import get_feedback_summary
    from Sylana_AI import MemoryDatabase, SylanaAgent
    use_new_memory = False

# Try to import optional multimodal functions
try:
    from Interaction_Interface.multimodal import transcribe_audio, process_image
    multimodal_available = True
except ImportError:
    multimodal_available = False
    print("⚠️  Multimodal features not available (CLIP/Whisper not installed)")

def display_startup_summary(memory_manager=None):
    """Display system startup information"""
    print("=" * 60)
    print("  SYLANA VESSEL - AI Companion System")
    if use_new_memory:
        print("  [SEMANTIC MEMORY ENABLED]")
    print("=" * 60)
    print()

    # Show configuration status
    print("System Configuration:")
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Database: {config.DB_PATH}")
    print(f"   Voice: {'Enabled' if config.ENABLE_VOICE else 'Disabled'}")
    print(f"   Fine-tuning: {'Enabled' if config.ENABLE_FINE_TUNING else 'Disabled'}")
    if use_new_memory:
        print(f"   Semantic Search: ENABLED ({config.EMBEDDING_MODEL})")
    print()

    # Show memory stats
    if use_new_memory and memory_manager:
        try:
            stats = memory_manager.get_stats()
            print("Memory System:")
            print(f"   Conversations: {stats['total_conversations']}")
            print(f"   Core Memories: {stats['total_core_memories']}")
            print(f"   FAISS Index: {stats['semantic_engine']['total_memories']} indexed")
            if stats['total_feedback'] > 0:
                print(f"   Avg Feedback: {stats['avg_feedback_score']}/5.0")
        except Exception as e:
            print(f"   Warning: {e}")
    else:
        # Old system
        try:
            from long_term_memory import build_index
            from adaptive_learning import get_feedback_summary
            index, texts = build_index()
            print(f"Memory System: {len(texts)} conversations indexed")
            avg_score, count = get_feedback_summary()
            if count > 0:
                print(f"   Feedback: {avg_score:.2f}/5.0 ({count} ratings)")
        except Exception as e:
            print(f"   Warning: {e}")

    print()
    print("=" * 60)
    print()


def start_conversation(agent):
    """Start text-based conversation with Sylana"""
    print("\n💬 Starting conversation mode (text-based)")
    print("   Type 'exit' to quit\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower().strip() in ["exit", "quit", "bye"]:
                print("\nSylana: Goodbye! I'll remember our conversation.")
                break

            if not user_input.strip():
                continue

            # Generate response using SylanaAgent
            response = agent.chat(user_input)
            print(f"Sylana: {response}\n")

        except KeyboardInterrupt:
            print("\n\nSylana: Goodbye! Conversation interrupted.")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            print("Let's try again...\n")


def multimodal_menu(agent):
    """Display multimodal options if available"""
    if not multimodal_available:
        print("⚠️  Multimodal features not available.")
        print("   Install dependencies: pip install transformers[torch] pillow")
        return start_conversation(agent)

    print("\n🎨 Choose an option:")
    print("   1. Test voice transcription")
    print("   2. Test image processing")
    print("   3. Start text conversation")
    print("   4. Exit")

    choice = input("\nYour choice (1-4): ").strip()

    if choice == "1":
        audio_file = input("Enter path to audio file: ")
        try:
            transcription = transcribe_audio(audio_file)
            print(f"\n🎤 Transcribed: {transcription}\n")
        except Exception as e:
            print(f"⚠️  Error: {e}")

    elif choice == "2":
        image_path = input("Enter path to image file: ")
        query_text = input("Enter query for image: ")
        try:
            probabilities = process_image(image_path, query_text)
            print(f"\n🖼️  Analysis: {probabilities}\n")
        except Exception as e:
            print(f"⚠️  Error: {e}")

    elif choice == "3":
        start_conversation(agent)

    elif choice == "4":
        print("\nGoodbye!")
        return

    else:
        print("Invalid choice. Starting conversation mode...")
        start_conversation(agent)


def main():
    """Main entry point for Sylana Vessel"""
    memory_manager = None

    # Initialize memory system
    if use_new_memory:
        print("\nInitializing enhanced memory system...")
        try:
            memory_manager = MemoryManager(config.DB_PATH)
            display_startup_summary(memory_manager)
            print("Semantic memory enabled!")
            print("For full semantic search features, use: python sylana_enhanced.py\n")
        except Exception as e:
            print(f"Warning: Could not initialize MemoryManager: {e}")
            print("Falling back to basic mode...\n")
            use_new_memory = False

    if not use_new_memory:
        display_startup_summary()

    # Initialize agent
    try:
        if use_new_memory and memory_manager:
            # Use new system (note: main.py uses simplified agent without full semantic search)
            # For full semantic features, use sylana_enhanced.py
            print("Note: Using simplified agent. For full semantic search, use sylana_enhanced.py")
            from Sylana_AI import MemoryDatabase, SylanaAgent
            db = MemoryDatabase(config.DB_PATH)
            agent = SylanaAgent(db)
        else:
            # Use original system
            db = MemoryDatabase(config.DB_PATH)
            agent = SylanaAgent(db)

        print("Sylana initialized successfully\n")
    except Exception as e:
        print(f"Failed to initialize Sylana: {e}")
        print("   Check your .env configuration and database setup")
        return 1

    try:
        # Check if multimodal features are requested
        if multimodal_available:
            multimodal_menu(agent)
        else:
            start_conversation(agent)

    finally:
        # Clean shutdown
        try:
            if use_new_memory and memory_manager:
                memory_manager.close()
            else:
                db.close()
            print("\nDatabase closed. Session saved.")
        except:
            pass

    return 0

if __name__ == "__main__":
    main()
