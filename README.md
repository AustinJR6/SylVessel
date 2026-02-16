# SYLANA VESSEL

**Complete emotionally intelligent AI companion - 100% Feature Complete** ðŸŽ‰

## Quick Start

### Full Model (13.5GB - Requires 15GB free disk space)
```bash
# Automatic setup (Windows)
quickstart.bat

# Automatic setup (Linux/Mac)
chmod +x quickstart.sh && ./quickstart.sh
```

### Quantized Model (6GB - For limited disk space / ancient laptops) â­ NEW!
```bash
# Windows - Optimized for CPU-only systems
quickstart_quantized.bat

# Manual setup
pip install llama-cpp-python
cp .env.quantized .env  # Add your HF_TOKEN
python memory/init_database.py
python sylana_quantized.py  # Quantized entry point
```

**See [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) or [QUANTIZED_SETUP_GUIDE.md](QUANTIZED_SETUP_GUIDE.md) for details**

## Features - ALL COMPLETE

**Semantic Memory System:**
- Supabase pgvector-powered semantic search
- Automatically recalls relevant past conversations
- Emotion-weighted importance scoring
- Core memory integration
- Recency-based boosting

**Intelligence & Learning:**
- DistilBERT emotion detection (6 states)
- Advanced prompt engineering
- Response validation & retry logic
- Feedback collection system
- Training data curation

**User Experience:**
- Voice interaction (optional)
- Multiple entry points
- Interactive memory explorer
- Real-time statistics
- Comprehensive logging

## Entry Points

Production entrypoint: `server.py` (FastAPI + Claude API + Supabase memory)


1. **`sylana.py`** â­ - Full model, all features (Requires 15GB disk space)
2. **`sylana_quantized.py`** ðŸš€ - Quantized model, all features (Only 6GB, CPU-optimized)
3. **`sylana_enhanced.py`** - Semantic memory focus
4. **`main.py`** - Simple interface
5. **`Sylana_AI.py`** - Direct AI core (legacy)

**Tools:**
- `tools/memory_explorer.py` - Explore conversation memory
- `quickstart.bat` / `quickstart.sh` - One-click setup (full model)
- `quickstart_quantized.bat` - One-click setup (quantized model)

## Status

âœ… **100% FEATURE COMPLETE - PRODUCTION READY**

- Phase 1: Security & Stability âœ…
- Phase 3: Semantic Memory âœ…
- Phase 4: Conversation Enhancement âœ…
- Phase 5: Learning & Feedback âœ…
- Polish: Tools & Documentation âœ…

**Total:** ~4,500 lines of code, 35+ files, 3,000+ lines docs

## Documentation

- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Get started in 5 minutes
- [CLOUD_RUN_DEPLOY.md](CLOUD_RUN_DEPLOY.md) - Deploy backend to Google Cloud Run
- [QUANTIZED_SETUP_GUIDE.md](QUANTIZED_SETUP_GUIDE.md) - Setup for limited disk space / ancient laptops
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Complete usage guide with examples
- [CHANGELOG.md](CHANGELOG.md) - Version history & all features
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Development roadmap
- [PROJECT_COMPLETION_REPORT.md](PROJECT_COMPLETION_REPORT.md) - What was built
- [DISK_SPACE_SOLUTION.md](DISK_SPACE_SOLUTION.md) - Solutions for low disk space
- [SECURITY_NOTICE.md](SECURITY_NOTICE.md) - Security information

**Created by Elias Ritt with Claude** ðŸš€
