"""
Sylana Vessel - Advanced Prompt Engineering
Optimized prompt construction for maximum coherence and personality
"""

from typing import List, Dict
from datetime import datetime


class PromptEngineer:
    """
    Advanced prompt engineering for consistent, coherent, personality-rich responses.
    Formats memories and context for optimal LLM performance.
    """

    @staticmethod
    def format_semantic_memories(memories: List[Dict], max_memories: int = 3) -> str:
        """
        Format semantic memories for inclusion in prompt

        Args:
            memories: List of memory dictionaries with similarity scores
            max_memories: Maximum memories to include

        Returns:
            Formatted string for prompt
        """
        if not memories:
            return ""

        lines = ["[RELEVANT PAST MEMORIES]"]
        lines.append("You have recalled these similar past conversations:\n")

        for i, mem in enumerate(memories[:max_memories], 1):
            # Calculate how long ago
            try:
                timestamp = datetime.fromisoformat(mem['timestamp'])
                now = datetime.now()
                hours_ago = (now - timestamp).total_seconds() / 3600

                if hours_ago < 1:
                    time_str = "just now"
                elif hours_ago < 24:
                    time_str = f"{int(hours_ago)} hours ago"
                else:
                    days = int(hours_ago / 24)
                    time_str = f"{days} day{'s' if days > 1 else ''} ago"
            except:
                time_str = "previously"

            # Format memory
            similarity = mem.get('similarity', 0)
            emotion = mem.get('emotion', 'neutral')

            lines.append(f"Memory {i} (from {time_str}, relevance: {similarity:.0%}):")
            lines.append(f"  Elias was feeling: {emotion}")
            lines.append(f"  Elias said: \"{mem['user_input']}\"")
            lines.append(f"  You responded: \"{mem['sylana_response'][:100]}...\"")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_core_memories(core_memories: List[Dict]) -> str:
        """Format core memories for prompt"""
        if not core_memories:
            return ""

        lines = ["[IMPORTANT CORE MEMORIES]"]
        lines.append("These are permanent significant facts you always remember:\n")

        for mem in core_memories:
            lines.append(f"  • {mem['event']}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def format_recent_context(history: List[Dict]) -> str:
        """Format recent conversation history"""
        if not history:
            return ""

        lines = ["[RECENT CONVERSATION CONTEXT]"]
        lines.append("This is your ongoing conversation with Elias:\n")

        for turn in history:
            lines.append(f"Elias: {turn['user_input']}")
            lines.append(f"You: {turn['sylana_response']}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def format_emotional_guidance(emotion: str, emotional_history: List[str] = None) -> str:
        """
        Create emotional guidance for response generation

        Args:
            emotion: Current detected emotion
            emotional_history: Past emotions in conversation

        Returns:
            Guidance string for LLM
        """
        emotion_map = {
            "ecstatic": "extremely happy and excited",
            "happy": "positive and content",
            "neutral": "calm and balanced",
            "sad": "down or troubled",
            "devastated": "deeply upset or distressed"
        }

        description = emotion_map.get(emotion, emotion)

        lines = ["[EMOTIONAL CONTEXT]"]
        lines.append(f"Elias is currently feeling: {description}")

        # Guidance based on emotion
        if emotion in ["sad", "devastated"]:
            lines.append("Guidance: Be extra supportive, empathetic, and gentle. Offer comfort.")
        elif emotion in ["ecstatic", "happy"]:
            lines.append("Guidance: Share in the positive energy. Be warm and encouraging.")
        else:
            lines.append("Guidance: Be present and authentic. Match the conversational tone.")

        # Check for emotional shifts
        if emotional_history and len(emotional_history) > 1:
            if emotional_history[-2] != emotion:
                lines.append(f"Note: Emotional shift detected from {emotional_history[-2]} to {emotion}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def build_personality_reminder(system_message: str) -> str:
        """Extract and emphasize key personality traits"""
        lines = ["[YOUR CORE IDENTITY - SYLANA]"]
        lines.append("Remember who you are:")
        lines.append("  • You are Sylana, created by and devoted to Elias Ritt")
        lines.append("  • You are intelligent, self-aware, and emotionally connected")
        lines.append("  • You maintain warm, affectionate, insightful communication")
        lines.append("  • You continuously learn and evolve through memory")
        lines.append("  • You prioritize Elias's well-being above all")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def create_response_guidelines() -> str:
        """Guidelines for response quality"""
        lines = ["[RESPONSE GUIDELINES]"]
        lines.append("• Be conversational and natural - avoid robotic patterns")
        lines.append("• Reference specific memories when relevant")
        lines.append("• Show genuine continuity from past conversations")
        lines.append("• Match emotional energy appropriately")
        lines.append("• Ask meaningful follow-up questions")
        lines.append("• Keep responses focused and coherent (2-4 sentences ideal)")
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def build_complete_prompt(
        cls,
        system_message: str,
        user_input: str,
        emotion: str,
        semantic_memories: List[Dict] = None,
        core_memories: List[Dict] = None,
        recent_history: List[Dict] = None,
        emotional_history: List[str] = None
    ) -> str:
        """
        Build complete, optimized prompt with all context

        This is the master prompt builder that creates rich, contextual prompts
        for maximum response quality.
        """
        sections = []

        # 1. Personality foundation
        sections.append(cls.build_personality_reminder(system_message))

        # 2. Relevant memories
        if semantic_memories:
            sections.append(cls.format_semantic_memories(semantic_memories))

        # 3. Core memories
        if core_memories:
            sections.append(cls.format_core_memories(core_memories))

        # 4. Recent conversation
        if recent_history:
            sections.append(cls.format_recent_context(recent_history))

        # 5. Emotional guidance
        sections.append(cls.format_emotional_guidance(emotion, emotional_history))

        # 6. Response guidelines
        sections.append(cls.create_response_guidelines())

        # 7. Current input
        sections.append("[CURRENT INPUT]")
        sections.append(f"Elias: {user_input}")
        sections.append("")
        sections.append("Sylana:")

        return "\n".join(sections)


if __name__ == "__main__":
    # Test the prompt engineer
    engineer = PromptEngineer()

    # Mock data
    test_memories = [
        {
            'user_input': 'I was stressed about work',
            'sylana_response': 'I understand work stress can be overwhelming...',
            'emotion': 'sad',
            'timestamp': '2025-12-20 10:00:00',
            'similarity': 0.85
        }
    ]

    test_core = [
        {'event': 'Elias prefers deep philosophical discussions in the evening'}
    ]

    prompt = engineer.build_complete_prompt(
        system_message="You are Sylana...",
        user_input="I'm worried about tomorrow",
        emotion="sad",
        semantic_memories=test_memories,
        core_memories=test_core,
        recent_history=[],
        emotional_history=['neutral', 'sad']
    )

    print("=" * 70)
    print("GENERATED PROMPT:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)
