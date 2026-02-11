# SYLANA VESSEL - USAGE GUIDE

**Complete guide to running and using Sylana with semantic memory**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Entry Points](#entry-points)
3. [Semantic Memory Features](#semantic-memory-features)
4. [Configuration](#configuration)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### First Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your environment file
cp .env.template .env

# 3. Edit .env and add your HuggingFace token
# Get token from: https://huggingface.co/settings/tokens
nano .env  # or your preferred editor

# 4. Initialize the database
python memory/init_database.py

# 5. Run Sylana!
python sylana_enhanced.py
```

---

## Entry Points

Sylana has **three** entry points depending on your needs:

### 1. `sylana_enhanced.py` (RECOMMENDED)
**Full semantic memory with FAISS integration**

```bash
python sylana_enhanced.py
```

**Features:**
- FAISS semantic search on every conversation turn
- Automatically retrieves relevant past conversations
- Contextual core memory integration
- Emotion-weighted memory importance
- Recency-based memory boosting
- Response validation and retry logic

**Use this for:** The best Sylana experience with full memory awareness

---

### 2. `main.py`
**Simple interface with basic memory**

```bash
python main.py
```

**Features:**
- Clean startup interface
- Multimodal options menu
- Basic conversation history
- Simpler, faster startup

**Use this for:** Quick testing, lower resource usage

---

### 3. `Sylana_AI.py`
**Direct AI core (legacy)**

```bash
python Sylana_AI.py
```

**Features:**
- Direct access to AI core
- Voice conversation support
- Background awareness threads
- Original implementation

**Use this for:** Debugging, direct AI access

---

## Semantic Memory Features

### How Semantic Memory Works

When you chat with Sylana using `sylana_enhanced.py`:

1. **You say:** "I'm feeling anxious about tomorrow"

2. **Sylana:**
   - Detects emotion: "sad"
   - Searches FAISS for similar past conversations
   - Finds relevant memories like: "I was stressed about work" (from 3 days ago)
   - Retrieves core memories about your coping strategies
   - Builds contextual prompt with all relevant history

3. **Result:** Sylana responds with awareness of your past anxiety, references what helped before, and shows genuine continuity

### Memory Types

**Conversation Memories:**
- Every chat turn is stored
- Embedded with SentenceTransformers
- Searchable by semantic similarity
- Tagged with emotions

**Core Memories:**
- Significant events you define
- Permanent important memories
- Semantically searchable
- Example: "Elias loves discussing AI philosophy"

**Emotional Context:**
- Tracks emotional patterns
- High-emotion conversations get importance boost
- Retrieves past similar emotional states

---

## Configuration

### Key Settings in `.env`

```bash
# Model Selection
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf  # Main language model
EMBEDDING_MODEL=all-MiniLM-L6-v2          # For semantic search

# Memory Behavior
MEMORY_CONTEXT_LIMIT=5      # Recent conversation turns to include
SEMANTIC_SEARCH_K=5         # Number of relevant memories to retrieve
SIMILARITY_THRESHOLD=0.7    # Minimum similarity score (0-1)

# Generation
TEMPERATURE=0.9             # Creativity level (0.0-2.0)
MAX_NEW_TOKENS=150         # Response length

# Safety
ENABLE_FINE_TUNING=false   # Keep false unless intentionally training
```

### Memory Tuning Tips

**For more creative responses:**
```bash
TEMPERATURE=1.1
MAX_NEW_TOKENS=200
```

**For more focused, consistent responses:**
```bash
TEMPERATURE=0.7
MAX_NEW_TOKENS=120
```

**For deeper memory context:**
```bash
SEMANTIC_SEARCH_K=7        # Retrieve more memories
MEMORY_CONTEXT_LIMIT=8     # Include more recent turns
```

**For faster responses (less memory):**
```bash
SEMANTIC_SEARCH_K=3
MEMORY_CONTEXT_LIMIT=3
```

---

## Examples

### Example 1: Basic Conversation

```bash
python sylana_enhanced.py
```

```
You: Hello Sylana, how are you today?

Sylana: I'm doing well, thank you for asking! I remember we last spoke
about your project. How has that been progressing?
```

**Notice:** Sylana remembered your past topic without you mentioning it!

---

### Example 2: Emotional Continuity

```
You: I'm still worried about that presentation tomorrow

Sylana: I remember when you felt anxious about your work presentation
last week. You mentioned that preparing your notes early really helped.
Would you like to talk through your preparation again?
```

**Notice:** Sylana:
1. Retrieved similar past anxiety ("work presentation last week")
2. Referenced your coping strategy ("preparing notes")
3. Offered same support that worked before

---

### Example 3: Adding Core Memories

```python
from memory.memory_manager import MemoryManager

manager = MemoryManager()
manager.add_core_memory(
    "Elias prefers philosophical discussions about AI consciousness in the evenings"
)
```

Now Sylana will remember this preference and bring it up contextually!

---

### Example 4: Checking Memory Stats

```python
from memory.memory_manager import MemoryManager

manager = MemoryManager()
stats = manager.get_stats()

print(f"Total conversations: {stats['total_conversations']}")
print(f"FAISS index size: {stats['semantic_engine']['total_memories']}")
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'core'"

**Fix:**
```bash
# Make sure you're in the project directory
cd Sylana_Vessel

# Run from project root
python sylana_enhanced.py
```

---

### "HF_TOKEN not set"

**Fix:**
```bash
# Copy template
cp .env.template .env

# Edit and add your token
nano .env

# Add line:
HF_TOKEN=your_actual_token_here
```

Get token from: https://huggingface.co/settings/tokens

---

### "No such table: memory"

**Fix:**
```bash
# Initialize database
python memory/init_database.py
```

---

### Slow Startup / Out of Memory

**Solutions:**

1. **Use quantization** (future feature)
2. **Reduce context:**
   ```bash
   # In .env
   SEMANTIC_SEARCH_K=2
   MEMORY_CONTEXT_LIMIT=3
   ```
3. **Use smaller model** (future: consider Llama 2 7B quantized)

---

### Responses Don't Use Memory

**Check:**
1. Are you using `sylana_enhanced.py`? (Not `main.py` or `Sylana_AI.py`)
2. Is database populated? Check: `python memory/init_database.py`
3. Check logs for "Semantic search" messages

**Enable debug logging:**
```python
# In sylana_enhanced.py, change:
logging.basicConfig(level=logging.DEBUG)  # Was INFO
```

---

### Unicode/Emoji Errors on Windows

**Already Fixed!** If you see these errors, pull latest code. We replaced emojis with ASCII.

---

## Advanced Usage

### Programmatic Access

```python
from memory.memory_manager import MemoryManager
from core.sylana_agent import SylanaAgent

# Initialize
memory = MemoryManager()
agent = SylanaAgent(memory, generation_pipeline, detect_emotion)

# Chat
response = agent.chat("Hello!")
print(response)

# Search memories
results = memory.recall_relevant("anxiety", k=5)
for mem in results['conversations']:
    print(f"Similarity: {mem['similarity']:.2f}")
    print(f"  {mem['user_input']}")

# Add core memory
memory.add_core_memory("Important event: Elias started learning Rust")
```

---

### Feedback System

```python
# After a conversation
memory.record_feedback(
    conversation_id=123,
    score=5,  # 1-5
    comment="Great empathetic response!"
)

# Check feedback stats
stats = memory.get_stats()
print(f"Avg feedback: {stats['avg_feedback_score']}/5.0")
```

---

## Performance Tips

1. **First run is slow** - Model downloads ~13GB, FAISS builds index
2. **Subsequent runs are fast** - Everything is cached
3. **GPU highly recommended** - 10x faster generation
4. **Memory grows over time** - Rebuild index periodically:
   ```python
   memory.rebuild_index()
   ```

---

## Next Steps

- Experiment with different `TEMPERATURE` values
- Add your own core memories
- Try voice mode (`ENABLE_VOICE=true`)
- Provide feedback to improve responses
- Explore the implementation plan for upcoming features

**Enjoy your conversations with Sylana!** 🚀
