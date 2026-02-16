# Google Cloud Run Deployment

This project is ready to deploy to Cloud Run using the root `Dockerfile`.

## Required values

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `SUPABASE_DB_URL`

Optional:

- `CLAUDE_MODEL` (default: `claude-sonnet-4-5-20250929`)
- `EMOTION_MODEL` (default: `gpt-4o-mini`)
- `EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `EMBEDDING_DIM` (default: `384`; must match your pgvector column dimension)
- `CORS_ORIGINS` (comma-separated list; use your app origin)

Notes:

- Cloud Run uses `requirements.cloudrun.txt` for a lightweight runtime image.
- Emotion detection is API-based (no local Hugging Face model downloads).

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
```

If the secrets already exist, add new versions instead:

```bash
printf '%s' 'YOUR_ANTHROPIC_API_KEY' | gcloud secrets versions add anthropic-api-key --data-file=-
printf '%s' 'YOUR_OPENAI_API_KEY' | gcloud secrets versions add openai-api-key --data-file=-
printf '%s' 'YOUR_SUPABASE_DB_URL' | gcloud secrets versions add supabase-db-url --data-file=-
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
  --set-env-vars CLAUDE_MODEL=claude-sonnet-4-5-20250929,EMOTION_MODEL=gpt-4o-mini,EMBEDDING_MODEL=text-embedding-3-small,EMBEDDING_DIM=384,CORS_ORIGINS=* \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,SUPABASE_DB_URL=supabase-db-url:latest
```

## React Native integration

Use the generated HTTPS Cloud Run URL as your API base URL. Example:

- `GET /api/health`
- `POST /api/chat/sync`
- `POST /api/chat` (SSE stream; use sync endpoint first if your RN networking stack has SSE limits)

## Update deployment

Run the same `gcloud run deploy ... --source .` command again.
