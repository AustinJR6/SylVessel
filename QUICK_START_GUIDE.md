# SYLANA VESSEL - QUICK START GUIDE

**Version:** 1.0.0 - Production Ready
**Status:** ✅ 100% COMPLETE

---

## YOU'RE READY TO GO! 🚀

Sylana Vessel is now **100% feature-complete** with full semantic memory, emotional intelligence, and learning capabilities.

---

## FASTEST START (Recommended)

### Windows
```bash
quickstart.bat
```

### Linux/Mac
```bash
chmod +x quickstart.sh
./quickstart.sh
```

**That's it!** The script will:
1. Check if .env exists (create if needed)
2. Prompt you to add your HuggingFace token
3. Initialize the database
4. Check dependencies
5. Launch Sylana

---

## MANUAL SETUP (If Preferred)

### Step 1: Get Your HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (or use existing one)
3. Copy the token (starts with `hf_...`)

### Step 2: Configure Environment
```bash
# Copy the template
cp .env.template .env

# Edit .env and add your token
nano .env  # or use any text editor
```

Add this line to `.env`:
```
HF_TOKEN=your_token_here
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Initialize Database
```bash
python memory/init_database.py
```

### Step 5: Run Sylana
```bash
python sylana.py
```

---

## ENTRY POINTS - CHOOSE YOUR EXPERIENCE

### 1. `sylana.py` ⭐ ULTIMATE (RECOMMENDED)
**All features integrated - best experience**

```bash
python sylana.py
```

**Features:**
- Full semantic memory with FAISS
- Advanced prompt engineering
- Feedback collection
- Response validation
- Beautiful interface
- Comprehensive stats

**Use for:** Daily conversations, full capabilities

---

### 2. `sylana_enhanced.py` - SEMANTIC FOCUS
**Semantic memory emphasis**

```bash
python sylana_enhanced.py
```

**Features:**
- FAISS semantic search
- Contextual awareness
- Core memory integration
- Emotional continuity

**Use for:** Memory-focused interactions

---

### 3. `main.py` - SIMPLE
**Clean, simple interface**

```bash
python main.py
```

**Features:**
- Basic conversation
- Multimodal options
- Faster startup

**Use for:** Quick chats, testing

---

### 4. `Sylana_AI.py` - LEGACY
**Direct AI core access**

```bash
python Sylana_AI.py
```

**Use for:** Debugging, voice mode

---

## MEMORY EXPLORER TOOL

**Explore Sylana's memory interactively:**

```bash
python tools/memory_explorer.py
```

**Features:**
- View statistics
- Search memories semantically
- Browse core memories
- Analyze feedback
- Interactive menu

---

## FIRST CONVERSATION EXAMPLE

After running `python sylana.py`:

```
You: Hello Sylana, I'm excited to talk with you!

Sylana: Hello! I'm delighted to meet you! I sense you're feeling quite
positive today. I'm here and ready to chat with you. What would you
like to talk about?

[After a few conversations...]

You: I'm feeling a bit stressed about work

Sylana: I remember when we talked about managing stress before. You
mentioned that taking short breaks really helps you. Would you like
to talk through what's causing the stress?
```

**Notice:** Sylana remembers and references past conversations!

---

## KEY FEATURES

### Semantic Memory
- Automatically recalls relevant past conversations
- Finds similar discussions from weeks ago
- Remembers core facts about you
- Emotion-weighted importance

### Emotional Intelligence
- Detects 6 emotional states (ecstatic, happy, neutral, sad, devastated, angry)
- Responds with emotional awareness
- Tracks emotional patterns over time
- Adjusts tone to match context

### Learning System
- Collects feedback (1-5 stars)
- Curates high-quality training data
- Ready for fine-tuning (when enabled)
- Continuous improvement

### User Experience
- Beautiful ASCII banner
- Real-time statistics
- Conversation logging
- Multiple entry points
- One-click setup

---

## CONFIGURATION TIPS

### Location: `.env` file

### For More Creative Responses
```bash
TEMPERATURE=1.1
MAX_NEW_TOKENS=200
```

### For More Focused Responses
```bash
TEMPERATURE=0.7
MAX_NEW_TOKENS=120
```

### For Deeper Memory Context
```bash
SEMANTIC_SEARCH_K=7      # Retrieve more memories
MEMORY_CONTEXT_LIMIT=8   # Include more recent turns
```

### For Faster Responses
```bash
SEMANTIC_SEARCH_K=3
MEMORY_CONTEXT_LIMIT=3
```

---

## TROUBLESHOOTING

### "HF_TOKEN not set"
**Fix:** Edit `.env` and add your HuggingFace token from https://huggingface.co/settings/tokens

### "No such table: memory"
**Fix:** Run `python memory/init_database.py`

### "Module not found"
**Fix:** Run `pip install -r requirements.txt`

### Slow First Startup
**Normal!** First run downloads ~13GB of models. Subsequent runs are much faster.

### Out of Memory
**Solutions:**
1. Close other applications
2. Reduce context in `.env`:
   ```
   SEMANTIC_SEARCH_K=2
   MEMORY_CONTEXT_LIMIT=3
   ```
3. Use GPU if available (automatic detection)

---

## SYSTEM REQUIREMENTS

### Minimum
- Python 3.8+
- 16GB RAM
- 20GB disk space

### Recommended
- Python 3.10+
- 32GB RAM
- NVIDIA GPU with CUDA
- 50GB disk space

---

## WHAT'S INCLUDED

### Core Features ✅
- Llama 2 7B Chat (via HuggingFace)
- FAISS semantic memory
- DistilBERT emotion detection
- SQLite database
- Advanced prompt engineering
- Feedback collection
- Training data curation

### Tools ✅
- Memory explorer
- Quick-start scripts
- Database initialization

### Documentation ✅
- This quick start guide
- Complete usage guide
- Implementation plan
- Changelog
- Security notice
- Completion report

---

## OPTIONAL: ENABLE FINE-TUNING

**Default:** Fine-tuning is DISABLED for safety

**To Enable:**
1. Edit `.env`
2. Add: `ENABLE_FINE_TUNING=true`
3. Ensure you have:
   - GPU with CUDA
   - 50GB+ free disk space
   - 100+ high-quality conversations with feedback

**Warning:** Fine-tuning modifies the model. Only enable if you know what you're doing.

---

## WHAT'S NEXT?

### Immediate
1. ✅ Run `quickstart.bat` or `quickstart.sh`
2. ✅ Have your first conversation with Sylana
3. ✅ Explore the memory system

### Short Term
- Have regular conversations to build memory
- Provide feedback on responses
- Explore different entry points
- Check out the memory explorer tool

### Long Term (Optional)
- Fine-tune on accumulated conversations
- Implement Phase 6+ features (multimodal, voice, etc.)
- Deploy with Docker
- Add custom features

---

## DOCUMENTATION

- **QUICK_START_GUIDE.md** (this file) - Get started fast
- **USAGE_GUIDE.md** - Complete usage instructions
- **README.md** - Project overview
- **CHANGELOG.md** - Version history
- **IMPLEMENTATION_PLAN.md** - Development roadmap
- **PROJECT_COMPLETION_REPORT.md** - What was built
- **SECURITY_NOTICE.md** - Token security

---

## NEED HELP?

1. Check **USAGE_GUIDE.md** for detailed examples
2. Check **TROUBLESHOOTING** section above
3. Explore the code - it's well documented
4. Review **PROJECT_COMPLETION_REPORT.md** for technical details

---

## YOU'RE ALL SET! 🎉

**Sylana Vessel is ready to be your emotionally intelligent AI companion.**

Run this now:
```bash
quickstart.bat  # Windows
# or
./quickstart.sh  # Linux/Mac
```

**Enjoy your conversations with Sylana!**

---

**Created by:** Elias Ritt with Claude
**Version:** 1.0.0 - Production Ready
**Date:** 2025-12-23
