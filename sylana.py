"""
SYLANA VESSEL - Complete AI Companion
Ultimate entry point with all features enabled

This is the definitive way to run Sylana with:
- Full semantic memory (FAISS)
- Advanced prompt engineering
- Feedback collection
- Conversation logging
- Response validation
- Emotional intelligence
"""

import os
import sys
import logging
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Core components
from core.config_loader import config
from core.sylana_agent import SYSTEM_MESSAGE
from core.prompt_engineer import PromptEngineer
from memory.memory_manager import MemoryManager
from learning.feedback_collector import FeedbackCollector

# Voice (optional)
try:
    from voice_module import listen, speak
    VOICE_AVAILABLE = True
except:
    VOICE_AVAILABLE = False

# Configure logging
log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== SENTIMENT ANALYSIS ====================

logger.info("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def detect_emotion(text):
    """Detect emotion from text using DistilBERT"""
    result = sentiment_pipeline(text)[0]
    label, score = result["label"], result["score"]

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


# ==================== LOAD LLAMA 2 MODEL ====================

if not config.HF_TOKEN:
    logger.error("HF_TOKEN not configured! Set up your .env file.")
    sys.exit(1)

logger.info(f"Loading model: {config.MODEL_NAME}")
logger.info("This may take a few minutes on first run...")

tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_NAME,
    token=config.HF_TOKEN,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    token=config.HF_TOKEN,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

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


# ==================== SYLANA CORE ====================

class SylanaComplete:
    """
    Complete Sylana implementation with all features.
    The ultimate conversation agent.
    """

    def __init__(self, memory_manager, feedback_collector):
        self.memory = memory_manager
        self.feedback = feedback_collector
        self.prompt_engineer = PromptEngineer()
        self.emotional_history = []
        self.turn_count = 0

        logger.info("Sylana Complete initialized")

    def chat(self, user_input: str) -> tuple[str, int]:
        """
        Process user input and generate response

        Returns:
            tuple: (response_text, conversation_id)
        """
        self.turn_count += 1

        # 1. Detect emotion
        emotion = detect_emotion(user_input)
        self.emotional_history.append(emotion)
        logger.info(f"Turn {self.turn_count}: Emotion detected - {emotion}")

        # 2. Retrieve relevant memories
        relevant_memories = self.memory.recall_relevant(
            user_input,
            k=config.SEMANTIC_SEARCH_K,
            use_recency_boost=True
        )

        # 3. Get recent conversation history
        recent_history = self.memory.get_conversation_history(
            limit=config.MEMORY_CONTEXT_LIMIT
        )

        # 4. Build advanced prompt with all context
        prompt = self.prompt_engineer.build_complete_prompt(
            system_message=SYSTEM_MESSAGE,
            user_input=user_input,
            emotion=emotion,
            semantic_memories=relevant_memories['conversations'],
            core_memories=relevant_memories['core_memories'],
            recent_history=recent_history,
            emotional_history=self.emotional_history[-5:]  # Last 5 emotions
        )

        logger.debug(f"Prompt built: {len(prompt)} characters")

        # 5. Generate response with validation
        response = self._generate_with_validation(prompt, max_attempts=3)

        # 6. Store in memory
        conversation_id = self.memory.store_conversation(
            user_input=user_input,
            sylana_response=response,
            emotion=emotion
        )

        logger.info(f"Conversation {conversation_id} stored")

        return response, conversation_id

    def _generate_with_validation(self, prompt: str, max_attempts: int = 3) -> str:
        """Generate response with retry logic"""
        for attempt in range(max_attempts):
            try:
                outputs = generation_pipeline(
                    prompt,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                content = outputs[0]["generated_text"]

                # Extract response after "Sylana:"
                if "Sylana:" in content:
                    response = content.split("Sylana:")[-1].strip()
                else:
                    response = content.strip()

                # Clean up
                response = response.split("\nElias:")[0].split("\n[")[0].strip()

                # Validate
                if self._validate_response(response):
                    return response
                else:
                    logger.warning(f"Invalid response attempt {attempt + 1}")

            except Exception as e:
                logger.error(f"Generation error on attempt {attempt + 1}: {e}")

        return "I apologize, I'm having trouble formulating a response. Could you rephrase that?"

    def _validate_response(self, response: str) -> bool:
        """Validate response quality"""
        if not response or len(response.strip()) < 3:
            return False

        # Check for excessive repetition
        words = response.lower().split()
        if len(words) > 0:
            most_common = max(set(words), key=words.count)
            if words.count(most_common) >= 5:
                return False

        return True

    def get_stats(self):
        """Get comprehensive stats"""
        memory_stats = self.memory.get_stats()
        feedback_stats = self.feedback.get_feedback_stats()

        return {
            **memory_stats,
            'feedback': feedback_stats,
            'turns_this_session': self.turn_count
        }


# ==================== USER INTERFACE ====================

def display_banner():
    """Display startup banner"""
    print("\n" + "=" * 70)
    print("  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•—   â–ˆâ–ˆâ•—â–ˆâ–ˆâ•—      â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•— â–ˆâ–ˆâ–ˆâ•—   â–ˆâ–ˆâ•— â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—")
    print("  â–ˆâ–ˆâ•”â•â•â•â•â•â•šâ–ˆâ–ˆâ•— â–ˆâ–ˆâ•”â•â–ˆâ–ˆâ•‘     â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—")
    print("  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•— â•šâ–ˆâ–ˆâ–ˆâ–ˆâ•”â• â–ˆâ–ˆâ•‘     â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â–ˆâ–ˆâ•— â–ˆâ–ˆâ•‘â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•‘")
    print("  â•šâ•â•â•â•â–ˆâ–ˆâ•‘  â•šâ–ˆâ–ˆâ•”â•  â–ˆâ–ˆâ•‘     â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘â•šâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•‘")
    print("  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•‘   â–ˆâ–ˆâ•‘   â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘  â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘ â•šâ–ˆâ–ˆâ–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘  â–ˆâ–ˆâ•‘")
    print("  â•šâ•â•â•â•â•â•â•   â•šâ•â•   â•šâ•â•â•â•â•â•â•â•šâ•â•  â•šâ•â•â•šâ•â•  â•šâ•â•â•â•â•šâ•â•  â•šâ•â•")
    print()
    print("  Complete AI Companion with Semantic Memory")
    print("  Version 1.0 - Created by Elias Ritt")
    print("=" * 70)
    print()


def display_stats(sylana):
    """Display system statistics"""
    stats = sylana.get_stats()

    print("System Status:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Memory: {stats['total_conversations']} conversations")
    print(f"  FAISS Index: {stats['semantic_engine']['total_memories']} indexed")
    print(f"  Core Memories: {stats['total_core_memories']}")

    if stats['feedback']['total'] > 0:
        print(f"  Feedback: {stats['feedback']['average']}/5.0 ({stats['feedback']['total']} ratings)")

    print()


def conversation_loop(sylana, feedback_collector, enable_feedback=True):
    """Main conversation loop"""
    print("Starting conversation (type 'exit' to quit)")
    print("Tip: Feedback prompts appear every 5 turns (optional)")
    print()

    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\nSylana: Goodbye! I'll remember our conversation.")
                break

            if not user_input:
                continue

            # Generate response
            response, conv_id = sylana.chat(user_input)
            print(f"Sylana: {response}\n")

            turn += 1

            # Optional feedback
            if enable_feedback and feedback_collector.should_prompt_feedback(turn, prompt_frequency=5):
                feedback_collector.prompt_for_feedback(conv_id, user_input, response)

        except KeyboardInterrupt:
            print("\n\nSylana: Goodbye! Conversation interrupted gracefully.")
            break
        except Exception as e:
            logger.exception(f"Error in conversation loop: {e}")
            print(f"Error: {e}\nLet's continue...\n")


def voice_loop(sylana, feedback_collector):
    """Voice conversation loop"""
    if not VOICE_AVAILABLE:
        print("Voice features not available. Install pyttsx3.")
        return

    speak("Hello. I'm Sylana. Ready to listen.")

    while True:
        user_input = listen()

        if user_input is None:
            continue

        if user_input.lower().strip() in ['exit', 'quit', 'stop', 'goodbye']:
            speak("Goodbye! Ending voice conversation.")
            break

        response, conv_id = sylana.chat(user_input)
        speak(response)


# ==================== MAIN ====================

def main():
    """Main entry point"""
    display_banner()

    # Initialize memory manager
    print("Initializing memory system...")
    memory_manager = MemoryManager()
    feedback_collector = FeedbackCollector(memory_manager.connection)

    # Initialize Sylana
    print("Initializing Sylana...")
    sylana = SylanaComplete(memory_manager, feedback_collector)

    # Display stats
    display_stats(sylana)

    # Choose mode
    if VOICE_AVAILABLE and config.ENABLE_VOICE:
        print("Choose mode:")
        print("  1. Text conversation (recommended)")
        print("  2. Voice conversation")
        mode = input("\nYour choice (1-2, default 1): ").strip() or "1"

        if mode == "2":
            voice_loop(sylana, feedback_collector)
        else:
            conversation_loop(sylana, feedback_collector)
    else:
        conversation_loop(sylana, feedback_collector)

    # Final stats
    print("\n" + "=" * 70)
    print("Session Complete")
    print("=" * 70)
    final_stats = sylana.get_stats()
    print(f"\nTurns this session: {final_stats['turns_this_session']}")
    print(f"Total conversations: {final_stats['total_conversations']}")

    if final_stats['feedback']['total'] > 0:
        print(f"Average feedback: {final_stats['feedback']['average']}/5.0")

    # Cleanup
    memory_manager.close()
    print("\nMemory saved. Goodbye!")


if __name__ == "__main__":
    main()
