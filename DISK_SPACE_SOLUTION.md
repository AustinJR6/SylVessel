# SYLANA VESSEL - DISK SPACE SOLUTION

**Problem:** Not enough disk space to download Llama 2 7B model (~13.5GB required)

**Your Current Free Space:** ~393MB

---

## SOLUTION OPTIONS

### Option 1: Free Up Disk Space (Best Experience)

**Required:** At least 15GB free on C: drive

**Quick Ways to Free Space:**

1. **Empty Recycle Bin**
   - Right-click Recycle Bin → Empty Recycle Bin

2. **Delete Temporary Files**
   - Press `Windows + R`
   - Type `temp` and press Enter
   - Select all files (Ctrl+A) and delete
   - Press `Windows + R` again
   - Type `%temp%` and press Enter
   - Select all files and delete

3. **Run Disk Cleanup**
   - Press `Windows + R`
   - Type `cleanmgr` and press Enter
   - Select C: drive
   - Check all boxes, especially "Temporary files" and "Downloads"
   - Click OK

4. **Delete Browser Cache**
   - Chrome: Settings → Privacy → Clear browsing data
   - Edge: Settings → Privacy → Choose what to clear

5. **Move Large Files**
   - Check Downloads folder
   - Check Videos/Pictures folders
   - Move to external drive or delete

6. **Uninstall Unused Programs**
   - Settings → Apps → Apps & features
   - Sort by size
   - Uninstall programs you don't use

**After freeing space:**
```bash
quickstart.bat
```

---

### Option 2: Use Lightweight Model (Immediate Solution)

**No disk space needed!** Switch to a smaller model that still works.

#### Step 1: Backup your current .env
```bash
copy .env .env.backup
```

#### Step 2: Use the lightweight configuration
```bash
copy .env.lightweight .env
```

#### Step 3: Run Sylana
```bash
python sylana.py
```

**What Changes:**
- Model: DistilGPT-2 (~500MB) instead of Llama 2 7B (~13GB)
- **All other features work the same:**
  - ✅ Semantic memory with FAISS
  - ✅ Emotion detection
  - ✅ Feedback collection
  - ✅ Memory explorer
  - ✅ All entry points

**Trade-offs:**
- Responses will be shorter and less sophisticated
- Still functional for testing the memory system
- Can upgrade to full model later when you have space

---

### Option 3: Use API-Based Model (Advanced)

Instead of downloading the model, use HuggingFace's API to run it in the cloud.

**Pros:**
- No disk space needed
- Full Llama 2 7B quality

**Cons:**
- Requires internet connection
- API calls cost money (after free tier)
- Slower response times

Let me know if you want instructions for this option.

---

## RECOMMENDATION

### For Testing Right Now:
**Use Option 2** (Lightweight Model)
- Works immediately
- Tests all features
- No disk space needed

### For Best Experience Later:
**Use Option 1** (Free Up Space)
- Full Llama 2 7B quality
- Better, more natural responses
- Worth the disk space

---

## QUICK COMMANDS

### Switch to Lightweight Model (Right Now):
```bash
copy .env.lightweight .env
python sylana.py
```

### Switch Back to Full Model (After freeing space):
```bash
copy .env.backup .env
python sylana.py
```

---

## DISK SPACE CHECK

Before running, check your free space:

**Windows:**
```bash
# In Command Prompt
wmic logicaldisk get size,freespace,caption
```

**You need:** At least 15GB free for full model

---

## NEED HELP?

If you're stuck, here are your options ranked by speed:

1. **Fastest:** Use lightweight model (Option 2) - works in 2 minutes
2. **Best:** Free up disk space (Option 1) - best experience, takes 30+ minutes
3. **Alternative:** Use API (Option 3) - requires setup

---

**Ready to proceed?** Choose Option 1 or 2 above!
