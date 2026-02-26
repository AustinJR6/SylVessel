# Google Cloud Run Deployment

This project is ready to deploy to Cloud Run using the root `Dockerfile`.

## Required values

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `SUPABASE_DB_URL`
- `BRAVE_SEARCH_API_KEY` (required if you want live web search in replies)
- `MEMORY_ENCRYPTION_KEY` (required for encrypted continuity payloads)

Optional:

- `CLAUDE_MODEL` (default: `claude-sonnet-4-20250514`)
- `EMOTION_MODEL` (default: `gpt-4o-mini`)
- `ENABLE_WEB_SEARCH` (default: `true`)
- `EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `EMBEDDING_DIM` (default: `384`; must match your pgvector column dimension)
- `MEMORY_DECAY_HALF_LIFE_DAYS` (default: `14`)
- `CONTINUITY_LOOKBACK_DAYS` (default: `30`)
- `MAX_CONTINUITY_ITEMS` (default: `8`)
- `CORS_ORIGINS` (comma-separated list; use your app origin)
- `CODE_EXEC_GCS_BUCKET` (bucket used by `/code/execute` for sandbox output artifacts)
- `CODE_EXEC_PYTHON_IMAGE` (default: `python:3.11-slim`)
- `CODE_EXEC_NODE_IMAGE` (default: `node:20-alpine`)
- `CODE_EXEC_BASH_IMAGE` (default: `bash:5.2`)
- `CODE_EXEC_CPUS` (default: `1`)
- `CODE_EXEC_MEMORY` (default: `768m`)

Notes:

- Cloud Run uses `requirements.cloudrun.txt` for a lightweight runtime image.
- Emotion detection is API-based (no local Hugging Face model downloads).
- `/code/execute` requires a Docker daemon on the backend host. Standard Cloud Run containers do not expose Docker-in-Docker by default.

## One-time setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

## Create secrets (recommended)

```bash
printf '%s' 'YOUR_ANTHROPIC_API_KEY' | gcloud secrets create anthropic-api-key --data-file=-
printf '%s' 'YOUR_OPENAI_API_KEY' | gcloud secrets create openai-api-key --data-file=-
printf '%s' 'YOUR_SUPABASE_DB_URL' | gcloud secrets create supabase-db-url --data-file=-
printf '%s' 'YOUR_BRAVE_SEARCH_API_KEY' | gcloud secrets create brave-search-api-key --data-file=-
printf '%s' 'YOUR_MEMORY_ENCRYPTION_KEY' | gcloud secrets create memory-encryption-key --data-file=-
```

If the secrets already exist, add new versions instead:

```bash
printf '%s' 'YOUR_ANTHROPIC_API_KEY' | gcloud secrets versions add anthropic-api-key --data-file=-
printf '%s' 'YOUR_OPENAI_API_KEY' | gcloud secrets versions add openai-api-key --data-file=-
printf '%s' 'YOUR_SUPABASE_DB_URL' | gcloud secrets versions add supabase-db-url --data-file=-
printf '%s' 'YOUR_BRAVE_SEARCH_API_KEY' | gcloud secrets versions add brave-search-api-key --data-file=-
printf '%s' 'YOUR_MEMORY_ENCRYPTION_KEY' | gcloud secrets versions add memory-encryption-key --data-file=-
```

## Deploy from source

```bash
gcloud run deploy sylana-vessel \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars CLAUDE_MODEL=claude-sonnet-4-20250514,EMOTION_MODEL=gpt-4o-mini,ENABLE_WEB_SEARCH=true,EMBEDDING_MODEL=text-embedding-3-small,EMBEDDING_DIM=384,MEMORY_DECAY_HALF_LIFE_DAYS=14,CONTINUITY_LOOKBACK_DAYS=30,MAX_CONTINUITY_ITEMS=8,CORS_ORIGINS=* \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,SUPABASE_DB_URL=supabase-db-url:latest,BRAVE_SEARCH_API_KEY=brave-search-api-key:latest,MEMORY_ENCRYPTION_KEY=memory-encryption-key:latest
```

## React Native integration

Use the generated HTTPS Cloud Run URL as your API base URL. Example:

- `GET /api/health`
- `POST /api/chat/sync`
- `POST /api/chat` (SSE stream; use sync endpoint first if your RN networking stack has SSE limits)

## Update deployment

Run the same `gcloud run deploy ... --source .` command again.
