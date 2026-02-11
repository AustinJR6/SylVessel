# 🚨 CRITICAL SECURITY NOTICE

## Exposed API Tokens - Immediate Action Required

### What Happened
HuggingFace API tokens were committed to this git repository in plaintext. These tokens are now considered **COMPROMISED** and must be revoked immediately.

### Exposed Tokens (Now Invalid)
The following tokens were found in the codebase:
- `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` (in Sylana_AI.py, fine_tuning.py)
- `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` (in continuous_training.py)

### Immediate Actions Required

#### 1. Revoke Compromised Tokens
Visit https://huggingface.co/settings/tokens and:
1. Log in to your HuggingFace account
2. Navigate to "Access Tokens"
3. Find and **DELETE** both tokens listed above
4. Confirm revocation

#### 2. Generate New Token
1. Create a new access token at https://huggingface.co/settings/tokens
2. Choose **Read** permissions (sufficient for model downloads)
3. Give it a descriptive name: "Sylana_Vessel_Local"
4. Copy the new token (you'll only see it once!)

#### 3. Configure Environment
1. Copy `.env.template` to `.env`:
   ```bash
   cp .env.template .env
   ```
2. Edit `.env` and paste your new token:
   ```
   HF_TOKEN=your_new_token_here
   ```
3. Save the file

#### 4. Verify .gitignore
Ensure `.env` is listed in `.gitignore` (already done in this update)

#### 5. Clean Git History (Optional but Recommended)
To remove tokens from git history:
```bash
# WARNING: This rewrites history. Only do this if you haven't pushed to a shared remote.
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch Sylana_AI.py fine_tuning.py continuous_training.py" \
  --prune-empty --tag-name-filter cat -- --all

# Or use BFG Repo-Cleaner (easier):
# java -jar bfg.jar --replace-text passwords.txt
```

### Why This Matters
Exposed API tokens can be used by anyone to:
- Access your HuggingFace account
- Download models using your quota
- Potentially incur costs if you upgrade to paid tiers
- Impersonate you in API requests

### Going Forward
**All secrets must now be stored in `.env` files:**
- ✅ `.env` is in `.gitignore`
- ✅ `.env.template` provides structure without secrets
- ✅ Code now loads secrets from environment variables
- ✅ Never commit `.env` to git

### Verification
After setup, verify your configuration:
```bash
python -c "from core.config_loader import config; print('✅ Config loaded successfully')"
```

---

**Status:** ⚠️ TOKENS COMPROMISED - ACTION REQUIRED
**Date:** 2025-12-23
**Responsibility:** Elias Ritt (repository owner)
