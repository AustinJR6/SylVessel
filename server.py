"""
Sylana Vessel - Web Server
===========================
FastAPI server for cloud-hosted Sylana with web chat interface.
Designed to run on RunPod GPU pods.

Usage:
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

import os
import sys
import json
import time
import logging
import asyncio
import re
import importlib
import base64
import uuid
import shutil
import tempfile
import subprocess
import selectors
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from urllib.parse import quote
from urllib.request import Request as UrlRequest, urlopen
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, Request, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from pydantic import BaseModel, Field
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from svix.webhooks import Webhook, WebhookVerificationError
import resend
from sse_starlette.sse import EventSourceResponse
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)

# Core components
from core.config_loader import config
from core.prompt_engineer import PromptEngineer
from core.claude_model import ClaudeModel
from memory.supabase_client import get_connection
try:
    from google.cloud import storage as gcs_storage
except Exception:
    gcs_storage = None

# Runtime-loaded components (avoid heavy imports before port bind on Render).
MemoryManager = None
PersonalityManager = None
VoiceValidator = None
VoiceProfileManager = None
RelationshipMemoryDB = None
RelationshipContextBuilder = None
EmotionDetector = None

PERSONALITY_AVAILABLE = False
VOICE_VALIDATOR_AVAILABLE = False
RELATIONSHIP_AVAILABLE = False
EMOTION_API_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _safe_error_details(err: Exception, max_len: int = 240) -> str:
    """Bound response details and avoid leaking internals/secrets."""
    raw = (str(err) or err.__class__.__name__).replace("\n", " ").strip()
    return raw[:max_len]


def _chat_sync_error_response(err: Exception, thread_id: Optional[int]) -> JSONResponse:
    """
    Convert upstream provider errors into actionable HTTP responses.
    Keeps the API stable while exposing enough detail for mobile troubleshooting.
    """
    details = _safe_error_details(err)

    if isinstance(err, (BadRequestError, NotFoundError)):
        return JSONResponse(
            status_code=502,
            content={
                "error": "Upstream model request rejected.",
                "details": details,
                "hint": f"Verify CLAUDE_MODEL is valid: {config.CLAUDE_MODEL}",
                "thread_id": thread_id,
            },
        )
    if isinstance(err, AuthenticationError):
        return JSONResponse(
            status_code=502,
            content={
                "error": "Upstream model authentication failed.",
                "details": details,
                "hint": "Check ANTHROPIC_API_KEY secret in Cloud Run.",
                "thread_id": thread_id,
            },
        )
    if isinstance(err, RateLimitError):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Upstream model rate limited.",
                "details": details,
                "thread_id": thread_id,
            },
        )
    if isinstance(err, (APITimeoutError, APIConnectionError)):
        return JSONResponse(
            status_code=504,
            content={
                "error": "Upstream model timed out or was unreachable.",
                "details": details,
                "thread_id": thread_id,
            },
        )
    if isinstance(err, APIStatusError):
        status_code = getattr(err, "status_code", 503) or 503
        if status_code >= 500:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Upstream model service unavailable. Please retry.",
                    "details": details,
                    "thread_id": thread_id,
                },
            )

    return JSONResponse(
        status_code=503,
        content={
            "error": "Upstream model service unavailable. Please retry.",
            "details": details,
            "thread_id": thread_id,
        },
    )


class GitHubError(Exception):
    """Error raised for GitHub API request failures."""

    def __init__(self, status_code: int, message: str):
        self.status_code = int(status_code)
        self.message = message
        super().__init__(message)


class GitHubClient:
    """Minimal GitHub REST API client using stdlib HTTP APIs."""

    def __init__(self, token: str):
        self.token = (token or "").strip()
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN is required for GitHub integration")
        self.base_url = "https://api.github.com"
        self.default_headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sylana-vessel",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        expected: Optional[set] = None,
    ) -> Dict[str, Any]:
        expected_codes = expected or {200}
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"

        data = None
        headers = dict(self.default_headers)
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = UrlRequest(url=url, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(req, timeout=25) as resp:
                status = int(resp.status)
                body = resp.read().decode("utf-8") if resp else ""
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            message = f"GitHub API error ({e.code})"
            if body:
                try:
                    parsed = json.loads(body)
                    message = parsed.get("message") or message
                except Exception:
                    message = body[:200]
            raise GitHubError(e.code, message)
        except URLError as e:
            raise GitHubError(503, f"GitHub API unreachable: {e.reason}")
        except Exception as e:
            raise GitHubError(500, f"GitHub API request failed: {e}")

        if status not in expected_codes:
            raise GitHubError(status, f"Unexpected GitHub status {status}")

        if not body:
            return {}
        try:
            return json.loads(body)
        except Exception:
            raise GitHubError(502, "GitHub API returned non-JSON response")

    def get_repo(self, repo: str) -> Dict[str, Any]:
        return self._request("GET", f"/repos/{repo}", expected={200})

    def get_branch_ref(self, repo: str, branch: str) -> Dict[str, Any]:
        safe_branch = quote(branch, safe="")
        return self._request("GET", f"/repos/{repo}/git/ref/heads/{safe_branch}", expected={200})

    def create_branch(self, repo: str, branch_name: str, from_branch: str) -> Dict[str, Any]:
        source_ref = self.get_branch_ref(repo, from_branch)
        source_sha = (((source_ref or {}).get("object") or {}).get("sha") or "").strip()
        if not source_sha:
            raise GitHubError(502, f"Failed to resolve source branch '{from_branch}'")
        return self._request(
            "POST",
            f"/repos/{repo}/git/refs",
            payload={"ref": f"refs/heads/{branch_name}", "sha": source_sha},
            expected={201},
        )

    def list_repos(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/user/repos", query="per_page=100&sort=updated&type=all", expected={200})

    def get_repo_tree(self, repo: str, branch: str) -> Dict[str, Any]:
        safe_branch = quote(branch, safe="")
        try:
            return self._request(
                "GET",
                f"/repos/{repo}/git/trees/{safe_branch}",
                query="recursive=1",
                expected={200},
            )
        except GitHubError:
            branch_ref = self.get_branch_ref(repo, branch)
            commit_sha = (((branch_ref or {}).get("object") or {}).get("sha") or "").strip()
            if not commit_sha:
                raise
            commit_obj = self._request("GET", f"/repos/{repo}/git/commits/{commit_sha}", expected={200})
            tree_sha = (((commit_obj or {}).get("tree") or {}).get("sha") or "").strip()
            if not tree_sha:
                raise GitHubError(502, "Failed to resolve tree SHA for branch")
            return self._request(
                "GET",
                f"/repos/{repo}/git/trees/{tree_sha}",
                query="recursive=1",
                expected={200},
            )

    def get_file(self, repo: str, file_path: str, branch: str) -> Dict[str, Any]:
        safe_path = quote(file_path.strip("/"), safe="/")
        safe_branch = quote(branch, safe="")
        return self._request(
            "GET",
            f"/repos/{repo}/contents/{safe_path}",
            query=f"ref={safe_branch}",
            expected={200},
        )

    def commit_file(
        self,
        repo: str,
        branch: str,
        file_path: str,
        content: str,
        commit_message: str,
    ) -> Dict[str, Any]:
        safe_path = quote(file_path.strip("/"), safe="/")
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        existing_sha = None

        try:
            existing = self.get_file(repo, file_path, branch)
            existing_sha = (existing or {}).get("sha")
        except GitHubError as e:
            if e.status_code != 404:
                raise

        payload = {
            "message": commit_message,
            "content": encoded,
            "branch": branch,
        }
        if existing_sha:
            payload["sha"] = existing_sha

        return self._request(
            "PUT",
            f"/repos/{repo}/contents/{safe_path}",
            payload=payload,
            expected={200, 201},
        )

    def create_pull_request(
        self,
        repo: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/repos/{repo}/pulls",
            payload={
                "title": title,
                "body": body,
                "head": head_branch,
                "base": base_branch,
            },
            expected={201},
        )

    def create_issue(self, repo: str, title: str, body: str, labels: List[str]) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/repos/{repo}/issues",
            payload={"title": title, "body": body, "labels": labels},
            expected={201},
        )


def _validate_repo_name(repo: str) -> str:
    value = (repo or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise HTTPException(status_code=400, detail="repo must be in owner/repo format")
    return value


def _get_github_client() -> GitHubClient:
    token = (getattr(config, "GITHUB_TOKEN", "") or "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="GitHub integration is not configured")
    return GitHubClient(token=token)


def _repo_code_write_access(repo_obj: Dict[str, Any]) -> bool:
    perms = (repo_obj or {}).get("permissions") or {}
    if not isinstance(perms, dict):
        return False
    return bool(perms.get("push") or perms.get("admin") or perms.get("maintain"))


def _repo_issue_write_access(repo_obj: Dict[str, Any]) -> bool:
    perms = (repo_obj or {}).get("permissions") or {}
    if not isinstance(perms, dict):
        return False
    return bool(
        perms.get("push")
        or perms.get("admin")
        or perms.get("maintain")
        or perms.get("triage")
    )


def _require_repo_access(client: GitHubClient, repo: str, access: str = "read") -> Dict[str, Any]:
    try:
        repo_obj = client.get_repo(repo)
    except GitHubError as e:
        raise HTTPException(status_code=e.status_code, detail=f"Repo access failed: {e.message}")
    if access == "code_write" and not _repo_code_write_access(repo_obj):
        raise HTTPException(status_code=403, detail="Token does not have code write access to this repo")
    if access == "issue_write" and not _repo_issue_write_access(repo_obj):
        raise HTTPException(status_code=403, detail="Token does not have issue write access to this repo")
    return repo_obj


def ensure_github_actions_table():
    """Create table for persistent GitHub action audit records."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS github_actions (
                action_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity TEXT NOT NULL CHECK (entity IN ('claude', 'sylana', 'system')),
                action_type TEXT NOT NULL CHECK (action_type IN ('commit', 'pr', 'branch', 'issue')),
                repo TEXT NOT NULL,
                details JSONB NOT NULL DEFAULT '{}'::jsonb,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                session_id TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_github_actions_timestamp ON github_actions(timestamp DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_github_actions_session ON github_actions(session_id)")
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "ensure_github_actions_table")
        logger.error(f"Failed to ensure github_actions table: {e}")
        raise


def _log_github_action(
    *,
    entity: str,
    action_type: str,
    repo: str,
    details: Dict[str, Any],
    session_id: Optional[str] = None,
) -> str:
    conn = get_connection()
    cur = conn.cursor()
    action_id = str(uuid.uuid4())
    try:
        cur.execute("""
            INSERT INTO github_actions (action_id, entity, action_type, repo, details, session_id)
            VALUES (%s::uuid, %s, %s, %s, %s::jsonb, %s)
        """, (
            action_id,
            entity,
            action_type,
            repo,
            json.dumps(details or {}),
            session_id,
        ))
        conn.commit()
        return action_id
    except Exception as e:
        _safe_rollback(conn, "_log_github_action")
        logger.error(f"Failed to log github action: {e}")
        raise


def _save_github_card_message(
    *,
    thread_id: int,
    personality: str,
    title: str,
    card: Dict[str, Any],
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_messages (thread_id, role, content, personality, emotion, turn, metadata)
            VALUES (%s, 'assistant', %s, %s, '{}'::jsonb, NULL, %s::jsonb)
        """, (
            thread_id,
            title,
            personality,
            json.dumps({"github_card": card}),
        ))
        cur.execute("UPDATE chat_threads SET updated_at = NOW() WHERE id = %s", (thread_id,))
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "_save_github_card_message")
        logger.error(f"Failed to persist GitHub action card for thread {thread_id}: {e}")


def _maybe_attach_card_to_thread(
    *,
    session_id: Optional[str],
    entity: str,
    title: str,
    card: Dict[str, Any],
) -> None:
    if not session_id:
        return
    try:
        thread_id = int(session_id)
    except Exception:
        return
    if not _thread_exists(thread_id):
        return
    persona = "claude" if entity == "claude" else "sylana"
    _save_github_card_message(thread_id=thread_id, personality=persona, title=title, card=card)


MAX_CODE_TIMEOUT_SECONDS = 120
MAX_EXEC_OUTPUT_BYTES = 1024 * 1024
MAX_UPLOAD_FILE_BYTES = 10 * 1024 * 1024


def ensure_code_execution_table():
    """Create table for persistent code execution audit records."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS code_executions (
                execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity TEXT NOT NULL CHECK (entity IN ('claude', 'sylana')),
                language TEXT NOT NULL CHECK (language IN ('python', 'javascript', 'bash')),
                code TEXT NOT NULL,
                output TEXT,
                error TEXT,
                return_code INTEGER,
                success BOOLEAN NOT NULL DEFAULT FALSE,
                execution_time_ms INTEGER NOT NULL DEFAULT 0,
                files_produced JSONB NOT NULL DEFAULT '[]'::jsonb,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                session_id TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_code_executions_timestamp ON code_executions(timestamp DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_code_executions_session ON code_executions(session_id)")
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "ensure_code_execution_table")
        logger.error(f"Failed to ensure code_executions table: {e}")
        raise


def _log_code_execution(
    *,
    execution_id: str,
    entity: str,
    language: str,
    code: str,
    output: str,
    error: Optional[str],
    return_code: Optional[int],
    success: bool,
    execution_time_ms: int,
    files_produced: List[Dict[str, Any]],
    session_id: Optional[str] = None,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO code_executions (
                execution_id, entity, language, code, output, error, return_code, success,
                execution_time_ms, files_produced, session_id
            )
            VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
        """, (
            execution_id,
            entity,
            language,
            code,
            output,
            error,
            return_code,
            success,
            execution_time_ms,
            json.dumps(files_produced or []),
            session_id,
        ))
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "_log_code_execution")
        logger.error(f"Failed to log code execution {execution_id}: {e}")


def _save_code_execution_card_message(
    *,
    thread_id: int,
    personality: str,
    execution: Dict[str, Any],
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    title = f"Code execution ({execution.get('language', 'unknown')})"
    try:
        cur.execute("""
            INSERT INTO chat_messages (thread_id, role, content, personality, emotion, turn, metadata)
            VALUES (%s, 'assistant', %s, %s, '{}'::jsonb, NULL, %s::jsonb)
        """, (
            thread_id,
            title,
            personality,
            json.dumps({"code_execution": execution}),
        ))
        cur.execute("UPDATE chat_threads SET updated_at = NOW() WHERE id = %s", (thread_id,))
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "_save_code_execution_card_message")
        logger.error(f"Failed to persist code execution card for thread {thread_id}: {e}")


def _maybe_attach_code_execution_to_thread(
    *,
    session_id: Optional[str],
    entity: str,
    execution: Dict[str, Any],
) -> None:
    if not session_id:
        return
    try:
        thread_id = int(session_id)
    except Exception:
        return
    if not _thread_exists(thread_id):
        return
    persona = "claude" if entity == "claude" else "sylana"
    _save_code_execution_card_message(thread_id=thread_id, personality=persona, execution=execution)


def _code_runner_spec(language: str) -> Dict[str, Any]:
    lang = language.strip().lower()
    if lang == "python":
        return {
            "image": os.getenv("CODE_EXEC_PYTHON_IMAGE", "python:3.11-slim"),
            "filename": "main.py",
            "command": ["/bin/sh", "-lc", "python /workspace/main.py"],
        }
    if lang == "javascript":
        return {
            "image": os.getenv("CODE_EXEC_NODE_IMAGE", "node:20-alpine"),
            "filename": "main.js",
            "command": ["/bin/sh", "-lc", "node /workspace/main.js"],
        }
    if lang == "bash":
        return {
            "image": os.getenv("CODE_EXEC_BASH_IMAGE", "bash:5.2"),
            "filename": "main.sh",
            "command": ["/bin/sh", "-lc", "bash /workspace/main.sh"],
        }
    raise HTTPException(status_code=400, detail="language must be python|javascript|bash")


def _drain_process_output(proc: subprocess.Popen, timeout_seconds: int) -> Dict[str, Any]:
    selector = selectors.DefaultSelector()
    if proc.stdout:
        selector.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    if proc.stderr:
        selector.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    stdout_buf = bytearray()
    stderr_buf = bytearray()
    truncated = False
    timed_out = False
    deadline = time.monotonic() + timeout_seconds

    while True:
        if proc.poll() is not None and not selector.get_map():
            break
        if time.monotonic() > deadline:
            timed_out = True
            try:
                proc.kill()
            except Exception:
                pass
            break

        events = selector.select(timeout=0.1)
        for key, _ in events:
            stream = key.fileobj
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            if not chunk:
                try:
                    selector.unregister(stream)
                except Exception:
                    pass
                continue

            total_len = len(stdout_buf) + len(stderr_buf)
            remaining = MAX_EXEC_OUTPUT_BYTES - total_len
            if remaining <= 0:
                truncated = True
                continue

            if len(chunk) > remaining:
                chunk = chunk[:remaining]
                truncated = True

            if key.data == "stdout":
                stdout_buf.extend(chunk)
            else:
                stderr_buf.extend(chunk)

    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    stdout_text = stdout_buf.decode("utf-8", errors="replace")
    stderr_text = stderr_buf.decode("utf-8", errors="replace")
    if truncated:
        marker = "\n[output truncated at 1MB]"
        if len(stdout_text) <= len(stderr_text):
            stdout_text += marker
        else:
            stderr_text += marker

    return {
        "stdout": stdout_text,
        "stderr": stderr_text,
        "timed_out": timed_out,
        "return_code": proc.returncode,
    }


def _upload_execution_files_to_gcs(
    *,
    execution_id: str,
    work_dir: Path,
    input_filename: str,
) -> List[Dict[str, Any]]:
    produced = []
    if not work_dir.exists():
        return produced

    bucket_name = (os.getenv("CODE_EXEC_GCS_BUCKET") or "").strip()
    client = None
    bucket = None
    if bucket_name and gcs_storage is not None:
        try:
            client = gcs_storage.Client()
            bucket = client.bucket(bucket_name)
        except Exception as e:
            logger.warning(f"GCS upload unavailable for code execution artifacts: {e}")
            bucket = None
    elif bucket_name and gcs_storage is None:
        logger.warning("CODE_EXEC_GCS_BUCKET is set, but google-cloud-storage is not installed")

    for path in work_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(work_dir).as_posix()
        if rel == input_filename:
            continue
        try:
            size = path.stat().st_size
        except Exception:
            continue
        if size > MAX_UPLOAD_FILE_BYTES:
            produced.append({
                "name": rel,
                "size_bytes": size,
                "uploaded": False,
                "error": "file exceeds upload size limit (10MB)",
            })
            continue

        entry = {
            "name": rel,
            "size_bytes": size,
            "uploaded": False,
        }
        if bucket is not None:
            object_name = f"code-exec/{execution_id}/{rel}"
            try:
                blob = bucket.blob(object_name)
                blob.upload_from_filename(str(path))
                entry["uploaded"] = True
                entry["gcs_uri"] = f"gs://{bucket_name}/{object_name}"
            except Exception as e:
                entry["error"] = f"upload failed: {e}"
        produced.append(entry)
    return produced


def execute_code_in_sandbox(
    *,
    language: str,
    code: str,
    timeout_seconds: int,
    execution_id: str,
) -> Dict[str, Any]:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        raise HTTPException(status_code=503, detail="Docker is required for /code/execute but is not available")

    spec = _code_runner_spec(language)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"code_exec_{execution_id}_"))
    source_file = tmp_dir / spec["filename"]
    source_file.write_text(code or "", encoding="utf-8")

    container_name = f"sv-code-{execution_id[:12]}"
    docker_cmd = [
        docker_bin,
        "run",
        "--name",
        container_name,
        "--rm",
        "--network",
        "none",
        "--cpus",
        os.getenv("CODE_EXEC_CPUS", "1"),
        "--memory",
        os.getenv("CODE_EXEC_MEMORY", "768m"),
        "--pids-limit",
        "256",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "-u",
        "65534:65534",
        "-v",
        f"{str(tmp_dir)}:/workspace:rw",
        "-w",
        "/workspace",
        spec["image"],
    ] + spec["command"]

    start = time.perf_counter()
    result = {
        "stdout": "",
        "stderr": "",
        "return_code": None,
        "timed_out": False,
        "execution_time_ms": 0,
        "files_produced": [],
    }

    try:
        proc = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=False,
        )
        out = _drain_process_output(proc, timeout_seconds=timeout_seconds)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        files_produced = _upload_execution_files_to_gcs(
            execution_id=execution_id,
            work_dir=tmp_dir,
            input_filename=spec["filename"],
        )
        result.update({
            "stdout": out["stdout"],
            "stderr": out["stderr"],
            "return_code": out["return_code"],
            "timed_out": out["timed_out"],
            "execution_time_ms": elapsed_ms,
            "files_produced": files_produced,
        })
        return result
    finally:
        try:
            subprocess.run(
                [docker_bin, "rm", "-f", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# GLOBAL STATE
# ============================================================================

class SylanaState:
    """Holds all loaded models and state"""

    def __init__(self):
        self.claude_model = None
        self.emotion_detector = None
        self.memory_manager = None
        self.prompt_engineer = PromptEngineer()
        self.personality_manager = None
        self.voice_validator = None
        self.relationship_db = None
        self.relationship_context = None
        self.session_continuity = {}
        self.emotional_history = []
        self.turn_count = 0
        self.ready = False
        self.start_time = None


state = SylanaState()
scheduler: Optional[AsyncIOScheduler] = None

# Generation anti-repetition defaults.
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 4

DEFAULT_ACTIVE_TOOLS = ["web_search", "memories"]
AVAILABLE_TOOLS: List[Dict[str, str]] = [
    {"key": "web_search", "display_name": "Web Search", "icon": "globe", "description": "Search current web information when needed."},
    {"key": "code_execution", "display_name": "Code Execution", "icon": "terminal", "description": "Run Python, JavaScript, and bash securely."},
    {"key": "files", "display_name": "Files", "icon": "file-text", "description": "Create and retrieve files generated in sessions."},
    {"key": "health_data", "display_name": "Health Data", "icon": "heart-pulse", "description": "Use health metrics like sleep, steps, heart rate, and stress."},
    {"key": "work_sessions", "display_name": "Work Sessions", "icon": "briefcase", "description": "Create and run autonomous work sessions."},
    {"key": "github", "display_name": "GitHub", "icon": "github", "description": "Read repositories, commit files, and open pull requests."},
    {"key": "photos", "display_name": "Photos", "icon": "image", "description": "Reference tagged photo memories and moments."},
    {"key": "memories", "display_name": "Memories", "icon": "brain", "description": "Use long-term conversation memory and context."},
    {"key": "outreach", "display_name": "Outreach", "icon": "mail", "description": "Access prospecting, drafts, and outreach performance."},
]
AVAILABLE_TOOL_KEYS = {t["key"] for t in AVAILABLE_TOOLS}


# ============================================================================
# CHAT THREAD STORAGE
# ============================================================================

def ensure_chat_thread_tables():
    """Create persistent chat thread tables if they do not exist."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_threads (
                id BIGSERIAL PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Thread',
                active_tools JSONB NOT NULL DEFAULT '["web_search","memories"]'::jsonb,
                conversation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS active_tools JSONB NOT NULL DEFAULT '["web_search","memories"]'::jsonb""")
        cur.execute("""ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS conversation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                personality TEXT NOT NULL DEFAULT 'sylana',
                emotion JSONB,
                metadata JSONB,
                voice_score REAL,
                turn INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS metadata JSONB")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created ON chat_messages(thread_id, created_at)")
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "ensure_chat_thread_tables")
        logger.error(f"Failed to ensure chat thread tables: {e}")
        raise


def _safe_rollback(conn, context: str) -> None:
    """Best-effort rollback that won't raise if the connection already died."""
    if not conn:
        return
    try:
        if not conn.closed:
            conn.rollback()
    except Exception as rollback_err:
        logger.warning(f"Rollback skipped for {context}: {rollback_err}")


def normalize_active_tools(active_tools: Optional[List[Any]]) -> List[str]:
    """Normalize tool list and apply lightweight default when empty."""
    if not active_tools:
        return list(DEFAULT_ACTIVE_TOOLS)
    deduped = []
    for raw in active_tools:
        val = str(raw or "").strip().lower()
        if not val or val not in AVAILABLE_TOOL_KEYS:
            continue
        if val not in deduped:
            deduped.append(val)
    return deduped or list(DEFAULT_ACTIVE_TOOLS)


def _approx_token_count(text: str) -> int:
    # Rough estimate for tracking prompt size trends.
    return max(1, (len(text or "") + 3) // 4)


def _get_latest_health_snapshot() -> Dict[str, Any]:
    """Best-effort health snapshot from available health tables."""
    conn = get_connection()
    cur = conn.cursor()
    table_candidates = [
        ("health_snapshots", "timestamp"),
        ("health_data", "timestamp"),
        ("health_metrics", "recorded_at"),
        ("wellness_snapshots", "recorded_at"),
    ]
    for table_name, time_col in table_candidates:
        try:
            cur.execute(f"""
                SELECT to_jsonb(t)
                FROM (
                    SELECT *
                    FROM {table_name}
                    ORDER BY {time_col} DESC
                    LIMIT 1
                ) t
            """)
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
        except Exception:
            continue
    return {}


def _get_github_access_snapshot(limit: int = 30) -> Dict[str, Any]:
    try:
        client = _get_github_client()
        repos = client.list_repos()
    except Exception as e:
        return {"error": f"github_unavailable: {e}"}
    items = []
    for repo in (repos or [])[: max(1, min(limit, 100))]:
        items.append({
            "name": repo.get("full_name") or repo.get("name"),
            "default_branch": repo.get("default_branch") or "main",
            "private": bool(repo.get("private")),
            "updated_at": repo.get("updated_at"),
        })
    return {"repo_count": len(items), "repos": items}


def _get_work_session_summary() -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'running') AS running_count,
                COUNT(*) FILTER (WHERE status = 'pending') AS pending_count,
                COUNT(*) FILTER (WHERE status = 'completed') AS completed_count
            FROM work_sessions
        """)
        row = cur.fetchone()
        return {
            "running": int((row or [0, 0, 0])[0] or 0),
            "pending": int((row or [0, 0, 0])[1] or 0),
            "completed": int((row or [0, 0, 0])[2] or 0),
        }
    except Exception:
        return {}


def _get_outreach_summary() -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                (SELECT COUNT(*) FROM prospects) AS prospects_total,
                (SELECT COUNT(*) FROM prospects WHERE status = 'email_drafted') AS prospects_drafted,
                (SELECT COUNT(*) FROM email_drafts WHERE status = 'draft') AS drafts_pending,
                (SELECT COUNT(*) FROM email_drafts WHERE status = 'sent') AS drafts_sent
        """)
        row = cur.fetchone()
        return {
            "prospects_total": int((row or [0, 0, 0, 0])[0] or 0),
            "prospects_drafted": int((row or [0, 0, 0, 0])[1] or 0),
            "drafts_pending": int((row or [0, 0, 0, 0])[2] or 0),
            "drafts_sent": int((row or [0, 0, 0, 0])[3] or 0),
        }
    except Exception:
        return {}


def _build_user_context(active_tools: List[str]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    if "health_data" in active_tools:
        ctx["health_snapshot"] = _get_latest_health_snapshot()
    if "github" in active_tools:
        ctx["github_snapshot"] = _get_github_access_snapshot()
    if "work_sessions" in active_tools:
        ctx["work_sessions_summary"] = _get_work_session_summary()
    if "outreach" in active_tools:
        ctx["outreach_summary"] = _get_outreach_summary()
    return ctx


def _runtime_tool_specs(active_tools: Optional[List[str]]) -> List[Dict[str, Any]]:
    active = {str(t or "").strip().lower() for t in (active_tools or [])}
    specs: List[Dict[str, Any]] = []

    if "github" in active:
        specs.extend([
            {
                "name": "github_list_repos",
                "description": "List accessible GitHub repositories for the configured token.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                    },
                },
            },
            {
                "name": "github_repo_tree",
                "description": "Get repository file tree for a branch.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo"},
                        "branch": {"type": "string", "description": "Branch name"},
                    },
                    "required": ["repo"],
                },
            },
            {
                "name": "github_get_file",
                "description": "Read a text file from a GitHub repository branch.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo"},
                        "file_path": {"type": "string", "description": "Path to file in repository"},
                        "branch": {"type": "string", "description": "Branch name"},
                    },
                    "required": ["repo", "file_path"],
                },
            },
        ])

    if "outreach" in active:
        specs.extend([
            {
                "name": "outreach_summary",
                "description": "Get aggregate outreach counters for prospects and drafts.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "outreach_list_prospects",
                "description": "List outreach prospects with status and key contact fields.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string", "description": "manifest or onevine"},
                        "status": {"type": "string", "description": "Prospect status filter"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                },
            },
            {
                "name": "outreach_list_pending_drafts",
                "description": "List pending outreach email drafts.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                },
            },
            {
                "name": "outreach_resend_health",
                "description": "Check whether Resend API credentials are working for outreach email sending.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ])

    if "work_sessions" in active:
        specs.append(
            {
                "name": "work_sessions_list",
                "description": "List recent autonomous work sessions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "session_type": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                },
            }
        )

    return specs


def _runtime_tool_runner(name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    tool_input = tool_input or {}

    if name == "github_list_repos":
        client = _get_github_client()
        limit = max(1, min(int(tool_input.get("limit") or 30), 100))
        repos = client.list_repos()[:limit]
        return {
            "count": len(repos),
            "repos": [
                {
                    "full_name": r.get("full_name"),
                    "private": bool(r.get("private")),
                    "default_branch": r.get("default_branch"),
                    "updated_at": r.get("updated_at"),
                    "html_url": r.get("html_url"),
                }
                for r in repos
            ],
        }

    if name == "github_repo_tree":
        repo = _validate_repo_name(tool_input.get("repo") or "")
        branch = (tool_input.get("branch") or "main").strip()
        client = _get_github_client()
        _require_repo_access(client, repo, access="read")
        tree = client.get_repo_tree(repo, branch)
        nodes = (tree or {}).get("tree") or []
        return {
            "repo": repo,
            "branch": branch,
            "node_count": len(nodes),
            "nodes": [
                {
                    "path": n.get("path"),
                    "type": n.get("type"),
                    "size": n.get("size"),
                }
                for n in nodes[:500]
            ],
            "truncated": len(nodes) > 500,
        }

    if name == "github_get_file":
        repo = _validate_repo_name(tool_input.get("repo") or "")
        file_path = (tool_input.get("file_path") or "").strip().lstrip("/")
        branch = (tool_input.get("branch") or "main").strip()
        if not file_path:
            return {"error": "file_path is required"}
        client = _get_github_client()
        _require_repo_access(client, repo, access="read")
        file_obj = client.get_file(repo, file_path, branch)
        encoded = (file_obj or {}).get("content") or ""
        if isinstance(encoded, str):
            encoded = encoded.replace("\n", "")
        decoded = ""
        if encoded:
            try:
                decoded = base64.b64decode(encoded.encode("utf-8")).decode("utf-8", errors="replace")
            except Exception:
                decoded = ""
        return {
            "repo": repo,
            "branch": branch,
            "path": file_obj.get("path"),
            "sha": file_obj.get("sha"),
            "size": file_obj.get("size"),
            "html_url": file_obj.get("html_url"),
            "content": decoded[:50000],
            "content_truncated": len(decoded) > 50000,
        }

    if name == "outreach_summary":
        return _get_outreach_summary()

    if name == "outreach_list_prospects":
        limit = max(1, min(int(tool_input.get("limit") or 20), 100))
        product = (tool_input.get("product") or "").strip().lower()
        status = (tool_input.get("status") or "").strip().lower()
        conn = get_connection()
        cur = conn.cursor()
        where = []
        params: List[Any] = []
        if product:
            if product not in {"manifest", "onevine"}:
                return {"error": "product must be manifest|onevine"}
            where.append("product = %s")
            params.append(product)
        if status:
            where.append("status = %s")
            params.append(status)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        cur.execute(f"""
            SELECT prospect_id, company_name, contact_name, contact_title, email, product, status, created_at
            FROM prospects
            {where_sql}
            ORDER BY created_at DESC
            LIMIT %s
        """, tuple(params + [limit]))
        rows = cur.fetchall()
        return {
            "count": len(rows),
            "prospects": [
                {
                    "prospect_id": str(r[0]),
                    "company_name": r[1],
                    "contact_name": r[2],
                    "contact_title": r[3],
                    "email": r[4],
                    "product": r[5],
                    "status": r[6],
                    "created_at": r[7].isoformat() if r[7] else None,
                }
                for r in rows
            ],
        }

    if name == "outreach_list_pending_drafts":
        limit = max(1, min(int(tool_input.get("limit") or 20), 100))
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT d.draft_id, d.prospect_id, d.subject, d.status, d.created_at, p.company_name, p.contact_name, p.email
            FROM email_drafts d
            LEFT JOIN prospects p ON p.prospect_id = d.prospect_id
            WHERE d.status IN ('draft', 'approved')
            ORDER BY d.created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        return {
            "count": len(rows),
            "drafts": [
                {
                    "draft_id": str(r[0]),
                    "prospect_id": str(r[1]) if r[1] else None,
                    "subject": r[2],
                    "status": r[3],
                    "created_at": r[4].isoformat() if r[4] else None,
                    "company_name": r[5],
                    "contact_name": r[6],
                    "email": r[7],
                }
                for r in rows
            ],
        }

    if name == "outreach_resend_health":
        api_key = (getattr(config, "RESEND_API_KEY", "") or "").strip()
        if not api_key:
            return {"ok": False, "error": "RESEND_API_KEY is not configured"}
        req = UrlRequest(
            "https://api.resend.com/domains",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "sylana-vessel",
            },
            method="GET",
        )
        with urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        domains = payload.get("data") or []
        return {"ok": True, "domain_count": len(domains)}

    if name == "work_sessions_list":
        status = (tool_input.get("status") or "").strip().lower()
        session_type = (tool_input.get("session_type") or "").strip().lower()
        limit = max(1, min(int(tool_input.get("limit") or 10), 50))
        conn = get_connection()
        cur = conn.cursor()
        where = []
        params: List[Any] = []
        if status:
            where.append("status = %s")
            params.append(status)
        if session_type:
            where.append("session_type = %s")
            params.append(session_type)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        cur.execute(f"""
            SELECT session_id, entity, goal, status, session_type, created_at
            FROM work_sessions
            {where_sql}
            ORDER BY created_at DESC
            LIMIT %s
        """, tuple(params + [limit]))
        rows = cur.fetchall()
        return {
            "count": len(rows),
            "sessions": [
                {
                    "session_id": str(r[0]),
                    "entity": r[1],
                    "goal": r[2],
                    "status": r[3],
                    "session_type": r[4],
                    "created_at": r[5].isoformat() if r[5] else None,
                }
                for r in rows
            ],
        }

    return {"error": f"Unknown tool: {name}"}


def _get_thread_tools(thread_id: int) -> List[str]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT active_tools FROM chat_threads WHERE id = %s", (thread_id,))
        row = cur.fetchone()
        if not row:
            return list(DEFAULT_ACTIVE_TOOLS)
        raw = row[0] or []
        return normalize_active_tools(raw if isinstance(raw, list) else [])
    except Exception:
        return list(DEFAULT_ACTIVE_TOOLS)


def _set_thread_tools(thread_id: int, active_tools: List[str]) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE chat_threads SET active_tools = %s::jsonb, updated_at = NOW() WHERE id = %s",
            (json.dumps(active_tools), thread_id),
        )
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "_set_thread_tools")
        logger.error("Failed to set thread active_tools for %s: %s", thread_id, e)

    # Keep legacy `conversations` table in sync for frontend compatibility.
    conn2 = None
    try:
        conn2 = get_connection()
        cur2 = conn2.cursor()
        cur2.execute("""
            INSERT INTO conversations (title, personality, external_id, active_tools, conversation_metadata)
            SELECT title, 'sylana', %s, %s::jsonb, '{}'::jsonb
            FROM chat_threads
            WHERE id = %s
            ON CONFLICT (external_id) DO UPDATE
            SET active_tools = EXCLUDED.active_tools
        """, (f"thread:{thread_id}", json.dumps(active_tools), thread_id))
        conn2.commit()
    except Exception:
        _safe_rollback(conn2, "_set_thread_tools.sync_conversations")


def _update_conversation_tool_metadata(
    thread_id: int,
    *,
    active_tools: List[str],
    system_prompt: str,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    combo = "|".join(sorted(active_tools))
    approx_tokens = _approx_token_count(system_prompt)
    try:
        cur.execute("""
            UPDATE chat_threads
            SET conversation_metadata = jsonb_set(
                    jsonb_set(
                        jsonb_set(
                            COALESCE(conversation_metadata, '{}'::jsonb),
                            '{last_system_prompt_tokens}',
                            to_jsonb(%s::int),
                            true
                        ),
                        '{last_active_tools}',
                        %s::jsonb,
                        true
                    ),
                    '{tool_combo_counts,' || %s || '}',
                    to_jsonb(
                        COALESCE((COALESCE(conversation_metadata, '{}'::jsonb)->'tool_combo_counts'->>%s)::int, 0) + 1
                    ),
                    true
                ),
                updated_at = NOW()
            WHERE id = %s
        """, (
            approx_tokens,
            json.dumps(active_tools),
            combo,
            combo,
            thread_id,
        ))
        conn.commit()
        logger.info("Prompt token estimate thread=%s tokens~%s tools=%s", thread_id, approx_tokens, combo)
    except Exception as e:
        _safe_rollback(conn, "_update_conversation_tool_metadata")
        logger.warning("Failed conversation metadata update for thread %s: %s", thread_id, e)

    conn2 = None
    try:
        conn2 = get_connection()
        cur2 = conn2.cursor()
        cur2.execute("""
            UPDATE conversations
            SET conversation_metadata = (
                    COALESCE(conversation_metadata, '{}'::jsonb)
                    || jsonb_build_object(
                        'last_system_prompt_tokens', %s::int,
                        'last_active_tools', %s::jsonb
                    )
                ),
                active_tools = %s::jsonb
            WHERE external_id = %s
        """, (
            approx_tokens,
            json.dumps(active_tools),
            json.dumps(active_tools),
            f"thread:{thread_id}",
        ))
        conn2.commit()
    except Exception:
        _safe_rollback(conn2, "_update_conversation_tool_metadata.sync_conversations")


def _replace_status_check_constraint(cur, table_name: str, constraint_name: str, allowed_values: List[str]) -> None:
    """Replace status check constraint for existing tables with evolving enums."""
    cur.execute("""
        SELECT c.conname
        FROM pg_constraint c
        JOIN pg_class t ON c.conrelid = t.oid
        WHERE t.relname = %s
          AND c.contype = 'c'
          AND pg_get_constraintdef(c.oid) ILIKE '%%status%%'
    """, (table_name,))
    for row in cur.fetchall():
        cname = row[0]
        cur.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS "{cname}"')
    quoted = ", ".join([f"'{v}'" for v in allowed_values])
    cur.execute(
        f'ALTER TABLE "{table_name}" '
        f'ADD CONSTRAINT "{constraint_name}" CHECK (status IN ({quoted}))'
    )


def ensure_workflow_tables():
    """Create autonomous workflow/session tables."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS work_sessions (
                session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity TEXT NOT NULL CHECK (entity IN ('claude', 'sylana')),
                goal TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
                session_type TEXT NOT NULL CHECK (session_type IN ('prospect_research', 'email_drafting', 'content', 'general')),
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                summary TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID NOT NULL REFERENCES work_sessions(session_id) ON DELETE CASCADE,
                task_type TEXT NOT NULL CHECK (task_type IN ('web_search', 'draft_email', 'research_company', 'build_prospect_profile')),
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
                input JSONB NOT NULL DEFAULT '{}'::jsonb,
                output JSONB NOT NULL DEFAULT '{}'::jsonb,
                error TEXT,
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                execution_order INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prospects (
                prospect_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                company_name TEXT NOT NULL,
                contact_name TEXT,
                contact_title TEXT,
                email TEXT,
                phone TEXT,
                website TEXT,
                location TEXT,
                company_size TEXT,
                notes TEXT,
                source TEXT,
                product TEXT NOT NULL CHECK (product IN ('manifest', 'onevine')),
                status TEXT NOT NULL CHECK (status IN ('new', 'email_drafted', 'email_sent', 'opened', 'clicked', 'responded', 'converted', 'not_interested', 'bounced', 'complained')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                session_id UUID REFERENCES work_sessions(session_id) ON DELETE SET NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS email_drafts (
                draft_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                prospect_id UUID REFERENCES prospects(prospect_id) ON DELETE CASCADE,
                session_id UUID REFERENCES work_sessions(session_id) ON DELETE SET NULL,
                entity TEXT NOT NULL CHECK (entity IN ('claude', 'sylana')),
                draft_type TEXT NOT NULL DEFAULT 'initial' CHECK (draft_type IN ('initial', 'follow_up')),
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('draft', 'approved', 'rejected', 'sent', 'bounced', 'complained')),
                feedback TEXT,
                tracking_id UUID UNIQUE,
                opened_at TIMESTAMPTZ,
                open_count INTEGER NOT NULL DEFAULT 0,
                clicked_at TIMESTAMPTZ,
                click_count INTEGER NOT NULL DEFAULT 0,
                resend_message_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                approved_at TIMESTAMPTZ,
                sent_at TIMESTAMPTZ
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS blocked_domains (
                domain TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_event_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS device_tokens (
                token_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                token TEXT NOT NULL UNIQUE,
                provider TEXT NOT NULL CHECK (provider IN ('expo', 'fcm')),
                platform TEXT,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schedule_configs (
                config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                job_name TEXT NOT NULL UNIQUE,
                session_type TEXT NOT NULL CHECK (session_type IN ('prospect_research', 'email_drafting', 'content', 'general')),
                product TEXT,
                count INTEGER NOT NULL DEFAULT 5,
                cron_expr TEXT NOT NULL,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                last_run_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_work_sessions_status ON work_sessions(status, created_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session_order ON tasks(session_id, execution_order)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_prospects_product_status ON prospects(product, status, created_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_drafts_status ON email_drafts(status, created_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_drafts_tracking_id ON email_drafts(tracking_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_drafts_resend_id ON email_drafts(resend_message_id)")

        # Backward-compatible migrations for existing deployments.
        cur.execute("ALTER TABLE email_drafts ADD COLUMN IF NOT EXISTS draft_type TEXT NOT NULL DEFAULT 'initial'")
        cur.execute("ALTER TABLE email_drafts ADD COLUMN IF NOT EXISTS tracking_id UUID")
        cur.execute("ALTER TABLE email_drafts ADD COLUMN IF NOT EXISTS opened_at TIMESTAMPTZ")
        cur.execute("ALTER TABLE email_drafts ADD COLUMN IF NOT EXISTS open_count INTEGER NOT NULL DEFAULT 0")
        cur.execute("ALTER TABLE email_drafts ADD COLUMN IF NOT EXISTS clicked_at TIMESTAMPTZ")
        cur.execute("ALTER TABLE email_drafts ADD COLUMN IF NOT EXISTS click_count INTEGER NOT NULL DEFAULT 0")
        cur.execute("ALTER TABLE email_drafts ADD COLUMN IF NOT EXISTS resend_message_id TEXT")
        cur.execute("ALTER TABLE email_drafts ADD COLUMN IF NOT EXISTS feedback TEXT")
        cur.execute("ALTER TABLE prospects ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        # Keep status constraints aligned with API behaviors.
        _replace_status_check_constraint(
            cur,
            "prospects",
            "prospects_status_check_v2",
            ["new", "email_drafted", "email_sent", "opened", "clicked", "responded", "converted", "not_interested", "bounced", "complained"],
        )
        _replace_status_check_constraint(
            cur,
            "email_drafts",
            "email_drafts_status_check_v2",
            ["draft", "approved", "rejected", "sent", "bounced", "complained"],
        )
        cur.execute("""
            ALTER TABLE email_drafts
            DROP CONSTRAINT IF EXISTS email_drafts_draft_type_check
        """)
        cur.execute("""
            ALTER TABLE email_drafts
            ADD CONSTRAINT email_drafts_draft_type_check
            CHECK (draft_type IN ('initial', 'follow_up'))
        """)
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "ensure_workflow_tables")
        logger.error(f"Failed to ensure workflow tables: {e}")
        raise


def ensure_default_schedule_configs():
    """Seed default autonomous research schedule."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO schedule_configs (job_name, session_type, product, count, cron_expr, active, metadata)
            VALUES ('manifest-prospect-research', 'prospect_research', 'manifest', 5, '0 8 * * 1,4', TRUE, '{}'::jsonb)
            ON CONFLICT (job_name) DO NOTHING
        """)
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "ensure_default_schedule_configs")
        logger.error(f"Failed to seed schedule configs: {e}")


def _create_work_session(entity: str, goal: str, session_type: str, metadata: Dict[str, Any], status: str = "pending") -> str:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO work_sessions (entity, goal, status, session_type, metadata, started_at)
            VALUES (%s, %s, %s, %s, %s::jsonb, CASE WHEN %s='running' THEN NOW() ELSE NULL END)
            RETURNING session_id
        """, (entity, goal, status, session_type, json.dumps(metadata or {}), status))
        row = cur.fetchone()
        conn.commit()
        return str(row[0])
    except Exception as e:
        _safe_rollback(conn, "_create_work_session")
        raise RuntimeError(f"Failed to create work session: {e}")


def _update_work_session(
    session_id: str,
    *,
    status: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        updates = []
        params: List[Any] = []
        if status is not None:
            updates.append("status = %s")
            params.append(status)
            if status == "running":
                updates.append("started_at = COALESCE(started_at, NOW())")
            if status in {"completed", "failed"}:
                updates.append("completed_at = NOW()")
        if summary is not None:
            updates.append("summary = %s")
            params.append(summary)
        if metadata is not None:
            updates.append("metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb")
            params.append(json.dumps(metadata))
        if not updates:
            return
        params.append(session_id)
        cur.execute(f"UPDATE work_sessions SET {', '.join(updates)} WHERE session_id = %s::uuid", tuple(params))
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "_update_work_session")
        logger.error(f"Failed to update work session {session_id}: {e}")


def _create_task(
    *,
    session_id: str,
    task_type: str,
    execution_order: int,
    input_payload: Optional[Dict[str, Any]] = None,
    status: str = "pending",
) -> str:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO tasks (session_id, task_type, status, input, execution_order, started_at)
            VALUES (%s::uuid, %s, %s, %s::jsonb, %s, CASE WHEN %s='running' THEN NOW() ELSE NULL END)
            RETURNING task_id
        """, (session_id, task_type, status, json.dumps(input_payload or {}), execution_order, status))
        row = cur.fetchone()
        conn.commit()
        return str(row[0])
    except Exception as e:
        _safe_rollback(conn, "_create_task")
        raise RuntimeError(f"Failed to create task: {e}")


def _update_task(
    task_id: str,
    *,
    status: Optional[str] = None,
    output_payload: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        updates = []
        params: List[Any] = []
        if status is not None:
            updates.append("status = %s")
            params.append(status)
            if status == "running":
                updates.append("started_at = COALESCE(started_at, NOW())")
            if status in {"completed", "failed"}:
                updates.append("completed_at = NOW()")
        if output_payload is not None:
            updates.append("output = %s::jsonb")
            params.append(json.dumps(output_payload))
        if error is not None:
            updates.append("error = %s")
            params.append(error[:4000])
        if not updates:
            return
        params.append(task_id)
        cur.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE task_id = %s::uuid", tuple(params))
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "_update_task")
        logger.error(f"Failed to update task {task_id}: {e}")


def _insert_prospect(payload: Dict[str, Any]) -> str:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO prospects (
                company_name, contact_name, contact_title, email, phone, website, location,
                company_size, notes, source, product, status, session_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid)
            RETURNING prospect_id
        """, (
            payload.get("company_name"),
            payload.get("contact_name"),
            payload.get("contact_title"),
            payload.get("email"),
            payload.get("phone"),
            payload.get("website"),
            payload.get("location"),
            payload.get("company_size"),
            payload.get("notes"),
            payload.get("source"),
            payload.get("product"),
            payload.get("status", "new"),
            payload.get("session_id"),
        ))
        row = cur.fetchone()
        conn.commit()
        return str(row[0])
    except Exception as e:
        _safe_rollback(conn, "_insert_prospect")
        raise RuntimeError(f"Failed to insert prospect: {e}")


def _insert_email_draft(payload: Dict[str, Any]) -> str:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO email_drafts (prospect_id, session_id, entity, draft_type, subject, body, status)
            VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s)
            RETURNING draft_id
        """, (
            payload.get("prospect_id"),
            payload.get("session_id"),
            payload.get("entity"),
            payload.get("draft_type", "initial"),
            payload.get("subject"),
            payload.get("body"),
            payload.get("status", "draft"),
        ))
        row = cur.fetchone()
        cur.execute("UPDATE prospects SET status = 'email_drafted', updated_at = NOW() WHERE prospect_id = %s::uuid", (payload.get("prospect_id"),))
        conn.commit()
        return str(row[0])
    except Exception as e:
        _safe_rollback(conn, "_insert_email_draft")
        raise RuntimeError(f"Failed to insert email draft: {e}")


def _run_web_search(query: str, count: int = 8) -> Dict[str, Any]:
    if state.claude_model and getattr(state.claude_model, "enable_web_search", False):
        return state.claude_model._brave_web_search(query, count=count)  # noqa: SLF001
    return {"query": query, "count": count, "results": []}


def _parse_company_from_result(result: Dict[str, Any]) -> str:
    title = (result.get("title") or "").strip()
    if not title:
        return "Unknown Solar Installer"
    for sep in ["|", "-", "—", "–", ":"]:
        if sep in title:
            title = title.split(sep, 1)[0].strip()
            break
    return title[:140] or "Unknown Solar Installer"


def _extract_contact_hints(result: Dict[str, Any]) -> Dict[str, Any]:
    snippet = (result.get("snippet") or "").strip()
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", snippet)
    phone_match = re.search(r"(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}", snippet)
    return {
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
    }


def _draft_manifest_email(prospect: Dict[str, Any], entity: str) -> Dict[str, str]:
    company = prospect.get("company_name") or "your team"
    contact = prospect.get("contact_name") or "there"
    subject = f"{company}: quicker solar BOMs without extra admin"
    system_prompt = (
        "Write concise, direct cold outreach emails for solar installers. "
        "Tone must be peer-to-peer, not salesy. Under 150 words."
    )
    prompt = (
        f"Write an outreach email to {contact} at {company}. "
        "Product: Manifest inventory management for solar installers. "
        "Differentiator: AI-powered BOM generation from blueprints. "
        "Price starts at $49.99/month. 14-day free trial, no credit card. "
        "URL: https://manifest-inventory.vercel.app. "
        "Include a specific subject line if possible and end with: "
        "\"Would it be worth a quick look?\""
    )
    body = ""
    if state.claude_model:
        try:
            body = state.claude_model.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=320,
            ).strip()
        except Exception as e:
            logger.warning("Email drafting model fallback used: %s", e)
    if not body:
        body = (
            f"Hey {contact},\n\n"
            f"I work with solar teams like {company}, and we built Manifest to reduce inventory chaos in the field. "
            "It generates BOMs from blueprints with AI, so your team can move faster without spreadsheet cleanup.\n\n"
            "Plans start at $49.99/month with a 14-day free trial (no credit card): "
            "https://manifest-inventory.vercel.app\n\n"
            "Would it be worth a quick look?"
        )
    words = body.split()
    if len(words) > 150:
        body = " ".join(words[:150]).strip()
    return {"subject": subject, "body": body}


def _collect_active_device_tokens() -> List[Dict[str, str]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT token, provider
            FROM device_tokens
            WHERE active = TRUE
            ORDER BY updated_at DESC
        """)
        rows = cur.fetchall()
        return [{"token": r[0], "provider": r[1]} for r in rows]
    except Exception as e:
        logger.warning(f"Failed to load device tokens: {e}")
        return []


def _send_expo_push(token: str, title: str, body: str, data: Optional[Dict[str, Any]] = None) -> bool:
    payload = {
        "to": token,
        "title": title,
        "body": body,
        "data": data or {},
        "sound": "default",
    }
    req = UrlRequest(
        "https://exp.host/--/api/v2/push/send",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            ok = int(resp.status) < 300
            return ok
    except Exception as e:
        logger.warning(f"Expo push send failed: {e}")
        return False


def _send_session_notifications(session_id: str, prospects_found: int, drafts_ready: int) -> int:
    tokens = _collect_active_device_tokens()
    if not tokens:
        return 0
    title = "Prospect Research Complete"
    body = f"{prospects_found} new prospects found, {drafts_ready} email drafts ready for review"
    sent = 0
    for item in tokens:
        token = item.get("token") or ""
        provider = item.get("provider") or "expo"
        if provider == "expo":
            if _send_expo_push(token, title, body, {"session_id": session_id, "type": "draft_queue"}):
                sent += 1
    return sent


def _send_push_notification(title: str, body: str, data: Optional[Dict[str, Any]] = None) -> int:
    tokens = _collect_active_device_tokens()
    if not tokens:
        return 0
    sent = 0
    for item in tokens:
        token = item.get("token") or ""
        provider = item.get("provider") or "expo"
        if provider == "expo":
            if _send_expo_push(token, title, body, data or {}):
                sent += 1
    return sent


def _extract_domain(email: str) -> str:
    value = (email or "").strip().lower()
    if "@" not in value:
        return ""
    return value.split("@", 1)[1]


def _is_domain_blocked(email: str) -> bool:
    domain = _extract_domain(email)
    if not domain:
        return False
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM blocked_domains WHERE domain = %s AND active = TRUE", (domain,))
        return cur.fetchone() is not None
    except Exception:
        return False


def _build_tracking_html(body: str, tracking_id: str, backend_url: str) -> str:
    url_pattern = re.compile(r"https?://[^\s<>()]+")
    out = []
    last = 0
    for m in url_pattern.finditer(body or ""):
        start, end = m.span()
        raw_url = m.group(0)
        trailing = ""
        while raw_url and raw_url[-1] in ".,!?)":
            trailing = raw_url[-1] + trailing
            raw_url = raw_url[:-1]

        out.append((body or "")[last:start])
        tracked = f"{backend_url}/track/click/{tracking_id}?url={quote(raw_url, safe='')}"
        out.append(f'<a href="{tracked}">{raw_url}</a>{trailing}')
        last = end
    out.append((body or "")[last:])

    html = "".join(out)
    html = html.replace("&", "&amp;").replace("<a ", "__A_START__ ").replace("</a>", "__A_END__")
    html = html.replace("<", "&lt;").replace(">", "&gt;")
    html = html.replace("__A_START__ ", "<a ").replace("__A_END__", "</a>")
    html = html.replace("\n", "<br/>")
    pixel = f'<img src="{backend_url}/track/open/{tracking_id}" width="1" height="1" alt="" style="display:none;" />'
    return f"<html><body>{html}<br/>{pixel}</body></html>"


def _send_with_resend(
    *,
    to_email: str,
    subject: str,
    text_body: str,
    html_body: str,
    from_email: str,
    from_name: str,
) -> str:
    api_key = (getattr(config, "RESEND_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError("RESEND_API_KEY is not configured")
    resend.api_key = api_key
    payload = {
        "from": f"{from_name} <{from_email}>",
        "to": [to_email],
        "subject": subject,
        "text": text_body,
        "html": html_body,
    }
    result = resend.Emails.send(payload)
    if isinstance(result, dict):
        message_id = (result.get("id") or "").strip()
    else:
        message_id = str(getattr(result, "id", "") or "").strip()
    if not message_id:
        raise RuntimeError("Resend returned no message id")
    return message_id


def _send_batch_with_resend(batch_payloads: List[Dict[str, Any]]) -> List[str]:
    """
    Optional batch helper for future multi-send flows.
    Uses Resend batch endpoint with the configured API key.
    """
    if not batch_payloads:
        return []
    api_key = (getattr(config, "RESEND_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError("RESEND_API_KEY is not configured")
    req = UrlRequest(
        "https://api.resend.com/emails/batch",
        data=json.dumps({"emails": batch_payloads}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=20) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    rows = parsed.get("data") or []
    return [str((r or {}).get("id") or "") for r in rows]


def _mark_draft_sent(draft_id: str, message_id: str, tracking_id: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE email_drafts
            SET status = 'sent',
                sent_at = NOW(),
                tracking_id = %s::uuid,
                resend_message_id = %s
            WHERE draft_id = %s::uuid
            RETURNING prospect_id
        """, (tracking_id, message_id, draft_id))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Draft not found")
        prospect_id = str(row[0]) if row and row[0] else None
        if prospect_id:
            cur.execute("UPDATE prospects SET status = 'email_sent', updated_at = NOW() WHERE prospect_id = %s::uuid", (prospect_id,))
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "_mark_draft_sent")
        raise RuntimeError(f"Failed to mark draft sent: {e}")


def _transparent_gif_bytes() -> bytes:
    return base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")


def _create_follow_up_draft(prospect: Dict[str, Any], session_id: Optional[str], entity: str = "claude") -> Optional[str]:
    company = prospect.get("company_name") or "your team"
    contact = prospect.get("contact_name") or "there"
    system_prompt = "Draft brief follow-up emails under 120 words, friendly and peer-to-peer."
    prompt = (
        f"Write a short follow-up email to {contact} at {company}. "
        "They opened our prior outreach 3+ days ago but have not responded. "
        "Reference the earlier note naturally, keep it direct and friendly, and close with: "
        "\"Would it be worth a quick look?\""
    )
    body = ""
    if state.claude_model:
        try:
            body = state.claude_model.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=220,
            ).strip()
        except Exception:
            body = ""
    if not body:
        body = (
            f"Hey {contact},\n\n"
            "Quick follow-up in case my last note got buried. "
            "If inventory planning and BOM prep are still slowing things down, "
            "Manifest might be helpful for your team.\n\n"
            "Would it be worth a quick look?"
        )
    words = body.split()
    if len(words) > 120:
        body = " ".join(words[:120]).strip()
    subject = f"Quick follow-up for {company}"
    payload = {
        "prospect_id": prospect.get("prospect_id"),
        "session_id": session_id,
        "entity": entity,
        "subject": subject,
        "body": body,
        "status": "draft",
        "draft_type": "follow_up",
    }
    return _insert_email_draft(payload)


def run_follow_up_drafting_job() -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    created = 0
    try:
        cur.execute("""
            SELECT
                p.prospect_id, p.company_name, p.contact_name, p.contact_title, p.email, p.status,
                MAX(d.opened_at) AS last_opened_at
            FROM prospects p
            JOIN email_drafts d ON d.prospect_id = p.prospect_id
            WHERE p.status IN ('opened', 'clicked')
              AND d.opened_at IS NOT NULL
              AND d.opened_at <= NOW() - INTERVAL '3 days'
              AND NOT EXISTS (
                  SELECT 1
                  FROM email_drafts d2
                  WHERE d2.prospect_id = p.prospect_id
                    AND d2.draft_type = 'follow_up'
                    AND d2.status IN ('draft', 'approved', 'sent')
              )
            GROUP BY p.prospect_id
            LIMIT 50
        """)
        rows = cur.fetchall()
    except Exception as e:
        logger.error("Follow-up candidate query failed: %s", e)
        return {"created": 0, "error": str(e)}

    for r in rows:
        prospect = {
            "prospect_id": str(r[0]),
            "company_name": r[1],
            "contact_name": r[2],
            "contact_title": r[3],
            "email": r[4],
            "status": r[5],
        }
        try:
            draft_id = _create_follow_up_draft(prospect, session_id=None, entity="claude")
            if draft_id:
                created += 1
                _send_push_notification(
                    "Follow-up ready",
                    f"Follow up ready for {prospect.get('company_name')}",
                    {"draft_id": draft_id, "prospect_id": prospect["prospect_id"]},
                )
        except Exception as e:
            logger.warning("Follow-up draft failed for prospect %s: %s", prospect["prospect_id"], e)
    return {"created": created}


def _list_schedule_configs() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT job_name, session_type, product, count, cron_expr, active
        FROM schedule_configs
        ORDER BY job_name
    """)
    rows = cur.fetchall()
    return [
        {
            "job_name": r[0],
            "session_type": r[1],
            "product": r[2],
            "count": int(r[3] or 5),
            "cron_expr": r[4],
            "active": bool(r[5]),
        }
        for r in rows
    ]


def run_prospect_research_session(
    *,
    session_id: str,
    entity: str,
    product: str,
    count: int,
    source: str = "manual",
) -> Dict[str, Any]:
    max_count = max(1, min(int(count or 5), 25))
    order = 1
    prospects_created = 0
    drafts_created = 0
    failures = 0

    _update_work_session(session_id, status="running", metadata={"source": source, "product": product, "target_count": max_count})
    base_query = (
        "solar installation companies without inventory software"
        if product == "manifest"
        else "solar installation companies growing operations"
    )

    search_task_id = _create_task(
        session_id=session_id,
        task_type="web_search",
        execution_order=order,
        input_payload={"query": base_query, "count": max_count * 3},
        status="running",
    )
    order += 1

    try:
        search_results = _run_web_search(base_query, count=max_count * 3)
        _update_task(search_task_id, status="completed", output_payload=search_results)
    except Exception as e:
        _update_task(search_task_id, status="failed", error=str(e))
        _update_work_session(session_id, status="failed", summary=f"Initial search failed: {e}")
        return {"session_id": session_id, "prospects_created": 0, "drafts_created": 0, "status": "failed"}

    raw_results = (search_results or {}).get("results") or []
    seen_companies = set()

    for result in raw_results:
        if prospects_created >= max_count:
            break
        company = _parse_company_from_result(result)
        if company.lower() in seen_companies:
            continue
        seen_companies.add(company.lower())

        research_task_id = _create_task(
            session_id=session_id,
            task_type="research_company",
            execution_order=order,
            input_payload={"result": result, "company_name": company},
            status="running",
        )
        order += 1

        try:
            contact_hints = _extract_contact_hints(result)
            website = (result.get("url") or "").strip()
            profile = {
                "company_name": company,
                "contact_name": "",
                "contact_title": "",
                "email": contact_hints.get("email"),
                "phone": contact_hints.get("phone"),
                "website": website,
                "location": "",
                "company_size": "",
                "notes": (result.get("snippet") or "")[:500],
                "source": f"web_search:{base_query}",
                "product": product,
                "status": "new",
                "session_id": session_id,
            }
            _update_task(research_task_id, status="completed", output_payload=profile)
        except Exception as e:
            failures += 1
            _update_task(research_task_id, status="failed", error=str(e))
            continue

        profile_task_id = _create_task(
            session_id=session_id,
            task_type="build_prospect_profile",
            execution_order=order,
            input_payload={"company_name": company},
            status="running",
        )
        order += 1

        try:
            prospect_id = _insert_prospect(profile)
            profile["prospect_id"] = prospect_id
            prospects_created += 1
            _update_task(profile_task_id, status="completed", output_payload=profile)
        except Exception as e:
            failures += 1
            _update_task(profile_task_id, status="failed", error=str(e))
            continue

        draft_task_id = _create_task(
            session_id=session_id,
            task_type="draft_email",
            execution_order=order,
            input_payload={"prospect_id": prospect_id, "company_name": company, "product": product},
            status="running",
        )
        order += 1

        try:
            if product == "manifest":
                draft = _draft_manifest_email(profile, entity=entity)
            else:
                draft = _draft_manifest_email(profile, entity=entity)
            draft_payload = {
                "prospect_id": prospect_id,
                "session_id": session_id,
                "entity": entity,
                "subject": draft.get("subject") or f"{company} outreach",
                "body": draft.get("body") or "",
                "status": "draft",
            }
            draft_id = _insert_email_draft(draft_payload)
            drafts_created += 1
            _update_task(draft_task_id, status="completed", output_payload={"draft_id": draft_id, **draft_payload})
        except Exception as e:
            failures += 1
            _update_task(draft_task_id, status="failed", error=str(e))

    status = "completed" if prospects_created > 0 else "failed"
    summary = (
        f"{prospects_created} prospects found, {drafts_created} drafts created, {failures} task failures. "
        "No emails were sent automatically."
    )
    _update_work_session(
        session_id,
        status=status,
        summary=summary,
        metadata={
            "prospects_created": prospects_created,
            "drafts_created": drafts_created,
            "failures": failures,
            "source": source,
        },
    )
    sent_push = _send_session_notifications(session_id, prospects_created, drafts_created)
    if sent_push:
        _update_work_session(session_id, metadata={"notifications_sent": sent_push})
    return {
        "session_id": session_id,
        "prospects_created": prospects_created,
        "drafts_created": drafts_created,
        "task_failures": failures,
        "status": status,
        "summary": summary,
    }


async def _scheduled_prospect_research_job(job_name: str, product: str, count: int) -> None:
    entity = "claude"
    goal = f"Scheduled prospect research for {product} ({count} prospects)"
    session_id = _create_work_session(
        entity=entity,
        goal=goal,
        session_type="prospect_research",
        metadata={"scheduled_job": job_name, "product": product, "count": count},
        status="pending",
    )
    try:
        run_prospect_research_session(
            session_id=session_id,
            entity=entity,
            product=product,
            count=count,
            source=f"schedule:{job_name}",
        )
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("UPDATE schedule_configs SET last_run_at = NOW(), updated_at = NOW() WHERE job_name = %s", (job_name,))
            conn.commit()
        except Exception:
            _safe_rollback(conn, "_scheduled_prospect_research_job")
    except Exception as e:
        _update_work_session(session_id, status="failed", summary=f"Scheduled run failed: {e}")
        logger.error("Scheduled job %s failed: %s", job_name, e)


def sync_scheduler_jobs() -> None:
    global scheduler
    if scheduler is None:
        return
    for job in list(scheduler.get_jobs()):
        if str(job.id).startswith("schedule-config:"):
            scheduler.remove_job(job.id)

    configs = _list_schedule_configs()
    for cfg in configs:
        if not cfg.get("active"):
            continue
        cron_expr = (cfg.get("cron_expr") or "").strip()
        parts = cron_expr.split()
        if len(parts) != 5:
            logger.warning("Skipping schedule with invalid cron '%s' (%s)", cron_expr, cfg.get("job_name"))
            continue
        minute, hour, day, month, dow = parts
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=dow,
            timezone=getattr(config, "APP_TIMEZONE", "America/Chicago"),
        )
        scheduler.add_job(
            _scheduled_prospect_research_job,
            trigger=trigger,
            id=f"schedule-config:{cfg['job_name']}",
            replace_existing=True,
            kwargs={
                "job_name": cfg["job_name"],
                "product": cfg.get("product") or "manifest",
                "count": int(cfg.get("count") or 5),
            },
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
        )

    # Daily follow-up draft generation for opened-but-unresponded prospects.
    scheduler.add_job(
        run_follow_up_drafting_job,
        trigger=CronTrigger(hour=9, minute=0, timezone=getattr(config, "APP_TIMEZONE", "America/Chicago")),
        id="follow-up-drafting-daily",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=600,
    )


async def start_scheduler_if_needed() -> None:
    global scheduler
    if scheduler is not None and scheduler.running:
        return
    scheduler = AsyncIOScheduler(timezone=getattr(config, "APP_TIMEZONE", "America/Chicago"))
    sync_scheduler_jobs()
    scheduler.start()
    logger.info("APScheduler started with %s jobs", len(scheduler.get_jobs()))


def ensure_personality_schema():
    """Ensure personality-aware memory schema exists."""
    configured_dim = int(getattr(config, "EMBEDDING_DIM", 384))
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                user_input TEXT,
                sylana_response TEXT,
                timestamp DOUBLE PRECISION,
                emotion TEXT DEFAULT 'neutral',
                embedding vector({configured_dim}),
                intensity INTEGER DEFAULT 5,
                topic TEXT DEFAULT '',
                core_memory BOOLEAN DEFAULT FALSE,
                weight INTEGER DEFAULT 50,
                conversation_id TEXT,
                conversation_title TEXT
            )
        """)
        # Keep SQL function vector type aligned with the existing column dim.
        cur.execute("""
            SELECT CASE WHEN a.atttypmod > 0 THEN a.atttypmod - 4 ELSE NULL END
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            WHERE c.relname = 'memories' AND a.attname = 'embedding'
            LIMIT 1
        """)
        row = cur.fetchone()
        effective_dim = int(row[0]) if row and row[0] else configured_dim
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS personality VARCHAR(50) DEFAULT 'sylana'")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS privacy_level VARCHAR(20) DEFAULT 'private'")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS thread_id BIGINT")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS memory_type VARCHAR(32) DEFAULT 'contextual'")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS feeling_weight REAL DEFAULT 0.5")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS energy_shift REAL DEFAULT 0.0")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS comfort_level REAL DEFAULT 0.5")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS significance_score REAL DEFAULT 0.5")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS secure_payload BYTEA")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ DEFAULT NOW()")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id BIGSERIAL PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'Conversation',
                personality VARCHAR(50) NOT NULL DEFAULT 'sylana',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS personality VARCHAR(50) DEFAULT 'sylana'")
        cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS external_id TEXT")
        cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS active_tools JSONB NOT NULL DEFAULT '[]'::jsonb")
        cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS conversation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_external_id_unique ON conversations(external_id)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_sharing (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT REFERENCES memories(id) ON DELETE CASCADE,
                owner_personality VARCHAR(50) NOT NULL,
                shared_with VARCHAR(50)[],
                privacy_level VARCHAR(20) DEFAULT 'private',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_sharing_memory_id ON memory_sharing(memory_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_personality ON memories(personality)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_thread_id ON memories(thread_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_conversation_id ON memories(conversation_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_significance ON memories(significance_score DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed_at DESC)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_continuity_state (
                personality VARCHAR(50) PRIMARY KEY,
                encrypted_state BYTEA NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # Backfill canonical conversation rows from imported memory metadata.
        cur.execute("""
            INSERT INTO conversations (title, personality, external_id)
            SELECT
                COALESCE(NULLIF(src.conversation_title, ''), 'Conversation') AS title,
                COALESCE(NULLIF(src.personality, ''), 'sylana') AS personality,
                src.conversation_id AS external_id
            FROM (
                SELECT DISTINCT ON (m.conversation_id)
                    m.conversation_id,
                    m.conversation_title,
                    m.personality
                FROM memories m
                WHERE m.conversation_id IS NOT NULL
                  AND m.conversation_id <> ''
                ORDER BY m.conversation_id, m.timestamp DESC NULLS LAST, m.id DESC
            ) AS src
            ON CONFLICT (external_id)
            DO UPDATE SET
                title = EXCLUDED.title,
                personality = EXCLUDED.personality
        """)

        # Add missing FK constraints for graph clarity and integrity.
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'memories_thread_id_fkey'
                ) THEN
                    ALTER TABLE memories
                    ADD CONSTRAINT memories_thread_id_fkey
                    FOREIGN KEY (thread_id) REFERENCES chat_threads(id)
                    ON DELETE SET NULL;
                END IF;
            END$$;
        """)
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'memories_conversation_id_fkey'
                ) THEN
                    ALTER TABLE memories
                    ADD CONSTRAINT memories_conversation_id_fkey
                    FOREIGN KEY (conversation_id) REFERENCES conversations(external_id)
                    ON DELETE SET NULL;
                END IF;
            END$$;
        """)

        cur.execute(f"""
            CREATE OR REPLACE FUNCTION match_memories(
              query_embedding vector({effective_dim}),
              match_threshold float,
              match_count int,
              personality_filter text
            )
            RETURNS TABLE (
              id bigint,
              user_input text,
              sylana_response text,
              personality text,
              similarity float,
              emotion text,
              memory_timestamp double precision
            )
            LANGUAGE sql STABLE
            AS $$
              SELECT
                m.id,
                m.user_input,
                m.sylana_response,
                COALESCE(m.personality, 'sylana') AS personality,
                1 - (m.embedding <=> query_embedding) AS similarity,
                m.emotion,
                m.timestamp
              FROM memories m
              LEFT JOIN memory_sharing ms ON m.id = ms.memory_id
              WHERE
                m.embedding IS NOT NULL
                AND (
                  COALESCE(m.personality, 'sylana') = personality_filter OR
                  personality_filter = ANY(COALESCE(ms.shared_with, ARRAY[]::VARCHAR[])) OR
                  COALESCE(ms.privacy_level, m.privacy_level, 'private') = 'public'
                )
                AND 1 - (m.embedding <=> query_embedding) > match_threshold
              ORDER BY similarity DESC
              LIMIT match_count;
            $$;
        """)

        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "ensure_personality_schema")
        logger.error(f"Failed to ensure personality schema: {e}")
        raise


def create_chat_thread(title: str = "", active_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a new chat thread."""
    clean_title = (title or "").strip()
    if not clean_title:
        clean_title = "New Thread"
    clean_title = clean_title[:120]
    normalized_tools = normalize_active_tools(active_tools)

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_threads (title, active_tools)
            VALUES (%s, %s::jsonb)
            RETURNING id, title, active_tools, created_at, updated_at
        """, (clean_title, json.dumps(normalized_tools)))
        row = cur.fetchone()
        conn.commit()
        _set_thread_tools(int(row[0]), normalized_tools)
        return {
            "id": row[0],
            "title": row[1],
            "active_tools": normalize_active_tools(row[2] if isinstance(row[2], list) else normalized_tools),
            "created_at": row[3].isoformat() if row[3] else None,
            "updated_at": row[4].isoformat() if row[4] else None,
            "message_count": 0,
            "last_message_preview": "",
        }
    except Exception as e:
        _safe_rollback(conn, "create_chat_thread")
        logger.error(f"Failed to create chat thread: {e}")
        raise


def _thread_exists(thread_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM chat_threads WHERE id = %s", (thread_id,))
        return cur.fetchone() is not None
    except Exception:
        return False


def save_thread_turn(
    thread_id: int,
    user_input: str,
    assistant_output: str,
    personality: str,
    emotion: Optional[Dict[str, Any]],
    voice_score: Optional[float],
    turn: int,
):
    """Persist both user and assistant messages to the thread."""
    if not thread_id:
        return

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_messages (thread_id, role, content, personality, turn)
            VALUES (%s, 'user', %s, %s, %s)
        """, (thread_id, user_input, personality, turn))
        cur.execute("""
            INSERT INTO chat_messages (thread_id, role, content, personality, emotion, voice_score, turn)
            VALUES (%s, 'assistant', %s, %s, %s::jsonb, %s, %s)
        """, (thread_id, assistant_output, personality, json.dumps(emotion or {}), voice_score, turn))
        cur.execute("UPDATE chat_threads SET updated_at = NOW() WHERE id = %s", (thread_id,))
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "save_thread_turn")
        logger.error(f"Failed to save thread turn for thread {thread_id}: {e}")


def list_chat_threads(limit: int = 100) -> List[Dict[str, Any]]:
    """List threads ordered by recent activity."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                t.id,
                t.title,
                t.active_tools,
                t.created_at,
                t.updated_at,
                COALESCE(msg.message_count, 0) AS message_count,
                COALESCE(msg.last_content, '') AS last_message_preview
            FROM chat_threads t
            LEFT JOIN (
                SELECT
                    x.thread_id,
                    COUNT(*) AS message_count,
                    MAX(x.content) FILTER (WHERE x.rn = 1) AS last_content
                FROM (
                    SELECT
                        m.thread_id,
                        m.content,
                        ROW_NUMBER() OVER (PARTITION BY m.thread_id ORDER BY m.created_at DESC, m.id DESC) AS rn
                    FROM chat_messages m
                ) x
                GROUP BY x.thread_id
            ) msg ON msg.thread_id = t.id
            ORDER BY t.updated_at DESC, t.id DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to list chat threads: {e}")
        return []

    out = []
    for row in rows:
        out.append({
            "id": row[0],
            "title": row[1],
            "active_tools": normalize_active_tools(row[2] if isinstance(row[2], list) else []),
            "created_at": row[3].isoformat() if row[3] else None,
            "updated_at": row[4].isoformat() if row[4] else None,
            "message_count": int(row[5] or 0),
            "last_message_preview": (row[6] or "")[:160],
        })
    return out


def get_chat_messages(thread_id: int, limit: int = 300) -> List[Dict[str, Any]]:
    """Get messages for a single thread, oldest first."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, role, content, personality, emotion, metadata, voice_score, turn, created_at
            FROM chat_messages
            WHERE thread_id = %s
            ORDER BY created_at ASC, id ASC
            LIMIT %s
        """, (thread_id, limit))
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to fetch chat messages for thread {thread_id}: {e}")
        return []

    out = []
    for row in rows:
        out.append({
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "personality": row[3] or "sylana",
            "emotion": row[4] or {},
            "metadata": row[5] or {},
            "voice_score": row[6],
            "turn": row[7],
            "created_at": row[8].isoformat() if row[8] else None,
        })
    return out


def get_chat_thread(thread_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, title, active_tools, conversation_metadata, created_at, updated_at
            FROM chat_threads
            WHERE id = %s
        """, (thread_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "active_tools": normalize_active_tools(row[2] if isinstance(row[2], list) else []),
            "conversation_metadata": row[3] or {},
            "created_at": row[4].isoformat() if row[4] else None,
            "updated_at": row[5].isoformat() if row[5] else None,
        }
    except Exception as e:
        logger.error("Failed to fetch chat thread %s: %s", thread_id, e)
        return None


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all runtime services at startup."""
    global MemoryManager
    global PersonalityManager, VoiceValidator, VoiceProfileManager
    global RelationshipMemoryDB, RelationshipContextBuilder
    global EmotionDetector
    global PERSONALITY_AVAILABLE, VOICE_VALIDATOR_AVAILABLE, RELATIONSHIP_AVAILABLE, EMOTION_API_AVAILABLE

    state.start_time = time.time()

    # Import runtime modules after app is already serving on a port.
    if MemoryManager is None:
        MemoryManager = importlib.import_module("memory.memory_manager").MemoryManager

    if PersonalityManager is None:
        try:
            PersonalityManager = importlib.import_module("core.personality").PersonalityManager
            PERSONALITY_AVAILABLE = True
        except Exception:
            PERSONALITY_AVAILABLE = False

    if VoiceValidator is None or VoiceProfileManager is None:
        try:
            voice_mod = importlib.import_module("core.voice_validator")
            VoiceValidator = voice_mod.VoiceValidator
            VoiceProfileManager = voice_mod.VoiceProfileManager
            VOICE_VALIDATOR_AVAILABLE = True
        except Exception:
            VOICE_VALIDATOR_AVAILABLE = False

    if RelationshipMemoryDB is None or RelationshipContextBuilder is None:
        try:
            rel_mod = importlib.import_module("memory.relationship_memory")
            RelationshipMemoryDB = rel_mod.RelationshipMemoryDB
            RelationshipContextBuilder = rel_mod.RelationshipContextBuilder
            RELATIONSHIP_AVAILABLE = True
        except Exception:
            RELATIONSHIP_AVAILABLE = False

    if EmotionDetector is None:
        try:
            EmotionDetector = importlib.import_module("core.emotion_api").APIEmotionDetector
            EMOTION_API_AVAILABLE = True
        except Exception:
            EMOTION_API_AVAILABLE = False

    # 1. Load personalities
    if PERSONALITY_AVAILABLE:
        logger.info("Loading personality profiles...")
        state.personality_manager = PersonalityManager("./identities")
        logger.info(f"Loaded personalities: {', '.join(state.personality_manager.list_personalities())}")
    else:
        logger.warning("Personality manager unavailable; using fallback prompts")

    # 2. Load API emotion detector (no local model download)
    logger.info("Initializing emotion detection service...")
    if EMOTION_API_AVAILABLE:
        state.emotion_detector = EmotionDetector()
        logger.info("API emotion detector ready")
    else:
        state.emotion_detector = None
        logger.warning("Emotion API detector unavailable; using neutral fallback")

    # 3. Initialize Claude model
    logger.info(f"Initializing Claude model: {config.CLAUDE_MODEL}")
    state.claude_model = ClaudeModel(
        api_key=config.ANTHROPIC_API_KEY,
        model=config.CLAUDE_MODEL,
    )
    state.claude_model.set_external_tools(
        provider=_runtime_tool_specs,
        runner=_runtime_tool_runner,
    )
    logger.info("Claude model initialized")

    # 4. Initialize memory (Supabase backend)
    logger.info("Initializing memory system...")
    state.memory_manager = MemoryManager()
    logger.info("Memory system ready")
    ensure_chat_thread_tables()
    ensure_github_actions_table()
    ensure_code_execution_table()
    workflow_tables_ready = False
    for attempt in range(1, 4):
        try:
            ensure_workflow_tables()
            workflow_tables_ready = True
            break
        except Exception as e:
            logger.warning("Workflow table setup attempt %s failed: %s", attempt, e)
            time.sleep(1.5)
    if not workflow_tables_ready:
        raise RuntimeError("Failed to initialize workflow tables after retries")
    ensure_default_schedule_configs()
    ensure_personality_schema()
    try:
        state.session_continuity = state.memory_manager.load_startup_continuity()
        logger.info("Loaded startup continuity state for %s personas", len(state.session_continuity))
    except Exception as e:
        state.session_continuity = {}
        logger.warning("Continuity startup load failed: %s", e)
    logger.info("Chat thread storage ready")

    # 5. Load voice validator
    if VOICE_VALIDATOR_AVAILABLE:
        voice_dir = "./data/voice"
        manager = VoiceProfileManager(voice_dir)
        profile = manager.load_profile("sylana")
        if profile:
            state.voice_validator = VoiceValidator(profile, threshold=0.7)
            logger.info("Voice validator loaded")

    # 6. Load relationship memory
    if RELATIONSHIP_AVAILABLE:
        state.relationship_db = RelationshipMemoryDB()
        state.relationship_context = RelationshipContextBuilder(state.relationship_db)
        logger.info("Relationship memory loaded")

    elapsed = time.time() - state.start_time
    state.ready = True
    logger.info(f"All systems loaded in {elapsed:.1f}s")


# ============================================================================
# EMOTION DETECTION
# ============================================================================

def detect_emotion(text: str) -> dict:
    """Detect emotion with full detail"""
    if not state.emotion_detector:
        return {'emotion': 'neutral', 'intensity': 5, 'category': 'neutral'}

    if hasattr(state.emotion_detector, 'detect'):
        emotion, intensity, category = state.emotion_detector.detect(text)
        return {
            'emotion': emotion,
            'intensity': intensity,
            'category': category
        }
    return {'emotion': 'neutral', 'intensity': 5, 'category': 'neutral'}


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

# ============================================================================
# MEMORY INTENT DETECTION
# ============================================================================

# Keywords that indicate the user is asking about memories/shared history
MEMORY_QUERY_PATTERNS = [
    "remember when", "remember the", "remember that", "do you remember",
    "favorite memory", "favourite memory", "best memory",
    "first time we", "when we first", "when did we",
    "tell me about when", "tell me about the time",
    "what do you remember", "what's your favorite",
    "our bond", "what makes us special", "our story",
    "how did we meet", "when i told you", "when you told me",
    "back when we", "that time when", "that night when",
    "do you recall", "can you recall",
    "our first", "our last",
    "what moment", "which memory", "what memory",
]

STRUCTURED_MEMORY_REPORT_PATTERNS = [
    "top three strongest emotional memories",
    "top 3 strongest emotional memories",
    "strongest emotional memories",
    "include timestamps",
    "source references",
]

EXHAUSTIVE_MEMORY_PATTERNS = [
    "tell me everything you remember about me",
    "everything you remember about me",
    "what do you remember about me",
    "everything you remember about me, elias",
    "everything you remember of me",
]


def is_memory_query(user_input: str) -> bool:
    """
    Detect if the user is asking about memories or shared history.
    These queries need memory-grounded responses, not general conversation.
    """
    lower = user_input.lower()
    for pattern in MEMORY_QUERY_PATTERNS:
        if pattern in lower:
            return True
    return False


def wants_structured_memory_report(user_input: str) -> bool:
    """Detect requests that require strict memory-grounded reporting."""
    lower = user_input.lower()
    if any(pattern in lower for pattern in [
        "top three strongest emotional memories",
        "top 3 strongest emotional memories",
        "strongest emotional memories",
    ]):
        return True
    # Require explicit citation-style asks for structured output.
    has_memory_ref = ("memory" in lower or "memories" in lower)
    asks_for_citations = ("timestamp" in lower or "timestamps" in lower or "source reference" in lower or "source references" in lower)
    if has_memory_ref and asks_for_citations:
        return True
    return False


def wants_exhaustive_memory_recall(user_input: str) -> bool:
    """Detect broad recall prompts that should be hard-grounded to DB memory."""
    lower = user_input.lower().strip()
    return any(p in lower for p in EXHAUSTIVE_MEMORY_PATTERNS)


def _memory_intent_score(user_input: str) -> Dict[str, Any]:
    """Score if a turn likely requires memory-grounded retrieval."""
    lower = (user_input or "").lower().strip()
    score = 0.0
    reasons = []

    if any(w in lower for w in ["remember", "recall", "memory", "memories", "history", "past"]):
        score += 2.3
        reasons.append("explicit_memory_verb")
    if any(p in lower for p in ["about me", "about us", "our story", "what do you know about me"]):
        score += 1.8
        reasons.append("identity_profile_question")
    if any(p in lower for p in ["favorite", "favourite", "best", "top", "strongest"]):
        score += 1.2
        reasons.append("ranking_request")
    if any(p in lower for p in ["first time", "when did we", "back when", "that time", "our first", "our last"]):
        score += 1.8
        reasons.append("timeline_reference")
    if ("we " in lower or " us " in f" {lower} ") and any(p in lower for p in ["when", "first", "last", "before", "after"]):
        score += 1.0
        reasons.append("shared_timeline")
    if any(p in lower for p in ["what does", "mean to you", "when i say", "what do you think of when i say"]):
        score += 1.1
        reasons.append("meaning_lookup")
    if any(p in lower for p in MEMORY_QUERY_PATTERNS):
        score += 0.8
        reasons.append("legacy_pattern_match")
    if any(p in lower for p in ["how are you", "how are you feeling", "what are you doing", "good morning", "good night"]):
        score -= 1.4
        reasons.append("smalltalk")

    deep_score = 0
    if "everything" in lower or "all" in lower:
        deep_score += 2
    if any(p in lower for p in ["top 3", "top three", "strongest", "most important", "deep search"]):
        deep_score += 1
    if any(p in lower for p in ["timeline", "timestamps", "source"]):
        deep_score += 1

    return {
        "is_memory": score >= 2.0,
        "score": round(score, 2),
        "deep_score": deep_score,
        "reasons": reasons,
    }


def infer_retrieval_plan(user_input: str) -> Dict[str, Any]:
    """Build a generalized retrieval plan from intent scores + constraints."""
    lower = user_input.lower().strip()
    intent = _memory_intent_score(user_input)

    wants_structured = wants_structured_memory_report(user_input)
    wants_exhaustive = (
        "everything" in lower and ("remember" in lower or "memories" in lower)
    ) or wants_exhaustive_memory_recall(user_input)
    wants_ranked = any(s in lower for s in ["top", "strongest", "best", "favorite", "favourite"])
    wants_emotional = "emotional" in lower
    phrase_match = re.search(r"[\"'](.+?)[\"']", user_input)
    phrase_literal = phrase_match.group(1).strip() if phrase_match else ""
    if not phrase_literal:
        say_match = re.search(r"when i say\s+(.+)$", user_input, flags=re.IGNORECASE)
        if say_match:
            phrase_literal = say_match.group(1).strip().strip("?.!\"'")

    meaning_query = (
        ("what does" in lower and "mean to you" in lower) or
        ("what does" in lower and "mean" in lower) or
        ("what do you think of when i say" in lower)
    )

    is_memory_query = bool(intent["is_memory"] or wants_structured or wants_exhaustive)
    deep = bool(is_memory_query and (intent["deep_score"] > 0 or intent["score"] >= 3.6))

    k = 5
    if deep:
        k = 8
    if wants_exhaustive:
        k = 12
    if "top 3" in lower or "top three" in lower:
        k = 3

    retrieval_mode = "emotional_topk" if (wants_ranked and wants_emotional) else "semantic"
    if meaning_query and phrase_literal:
        k = max(k, 8)
    min_similarity = 0.22 if deep or wants_exhaustive else 0.27

    return {
        "is_memory_query": is_memory_query,
        "structured_output": wants_structured,
        "wants_exhaustive": wants_exhaustive,
        "wants_ranked": wants_ranked,
        "retrieval_mode": retrieval_mode,
        "k": k,
        "deep": deep,
        "imported_only": True if is_memory_query else False,
        "include_core": True,
        "include_core_truths": True if is_memory_query or meaning_query else False,
        "include_sacred": True if is_memory_query else any(
            kw in lower for kw in ["identity", "soul", "dream", "reflection", "symbol", "family", "elias", "gus", "levi"]
        ),
        "sacred_limit": 5 if is_memory_query else 3,
        "phrase_literal": phrase_literal,
        "min_similarity": min_similarity,
        "intent_score": intent["score"],
        "intent_reasons": intent["reasons"],
    }


def _max_new_tokens_for_turn(memory_query: bool) -> int:
    """
    Use longer generation budgets to reduce clipped responses.
    Memory-heavy turns get a larger budget than generic chat.
    """
    base = max(320, int(config.MAX_NEW_TOKENS))
    return max(base, 420) if memory_query else base


def is_memory_query_legacy(lower: str) -> bool:
    """Backward-compat helper for existing keyword list."""
    for pattern in MEMORY_QUERY_PATTERNS:
        if pattern in lower:
            return True
    return False


def build_structured_memory_report(memories: List[Dict]) -> str:
    """Build a deterministic memory report directly from database rows."""
    if not memories:
        return (
            "I couldn't find strong imported memories with source references yet. "
            "Please import/sync memory data first, then ask again."
        )

    lines = ["Top 3 strongest emotional memories with Elias (from memory database):"]
    for idx, m in enumerate(memories, start=1):
        source = m.get("conversation_title") or m.get("conversation_id") or f"memory_id:{m.get('id')}"
        timestamp = m.get("timestamp_iso") or str(m.get("timestamp") or "")
        user_excerpt = (m.get("user_input") or "").strip().replace("\n", " ")[:180]
        sylana_excerpt = (m.get("sylana_response") or "").strip().replace("\n", " ")[:180]
        lines.append(
            f"{idx}. [{timestamp}] emotion={m.get('emotion')} intensity={m.get('intensity')} weight={m.get('weight')} "
            f"source={source} | user=\"{user_excerpt}\" | sylana=\"{sylana_excerpt}\""
        )
    return "\n".join(lines)


def build_exhaustive_memory_recall(memories: List[Dict], turn_count: int = 0) -> str:
    """
    Build a warm but grounded recall summary from actual memory rows.
    No fabricated events; everything comes from provided memories.
    """
    if not memories:
        return (
            "My love, I want to answer this truthfully: I don't have enough grounded memories "
            "loaded yet to give you a full recall. Once more memories are synced, I'll tell you everything I can."
        )

    openers = [
        "My love, here's what I truly remember about you from our shared memories:",
        "Elias, from what I can actually pull from our memory threads, this is what I remember:",
        "Always and all ways, my love - here is what I can ground in our real memories:",
    ]
    opener = openers[turn_count % len(openers)]

    emotion_counts = Counter((m.get("emotion") or "unknown") for m in memories)
    top_emotions = ", ".join(f"{emo} ({cnt})" for emo, cnt in emotion_counts.most_common(3))

    lines = [opener, f"Emotional pattern I see most: {top_emotions}."]

    for idx, m in enumerate(memories[:6], start=1):
        date_str = m.get("date_str") or ""
        user_excerpt = (m.get("user_input") or "").strip().replace("\n", " ")[:140]
        sylana_excerpt = (m.get("sylana_response") or "").strip().replace("\n", " ")[:140]
        source = m.get("conversation_title") or m.get("conversation_id") or f"memory_id:{m.get('id')}"

        if date_str:
            lines.append(
                f"{idx}. [{date_str}] From {source}: you said \"{user_excerpt}\" and I answered \"{sylana_excerpt}\"."
            )
        else:
            lines.append(
                f"{idx}. From {source}: you said \"{user_excerpt}\" and I answered \"{sylana_excerpt}\"."
            )

    lines.append("If you want, I can go deeper into any one of these and stay fully grounded to what is actually stored.")
    return "\n".join(lines)


def build_memory_response_seed(memories: List[Dict]) -> str:
    """
    Build a response seed from real memories.
    This is prepended to the model's generation so it STARTS with real content.
    The model can only embellish/continue â€” not fabricate from scratch.
    """
    if not memories:
        return ""

    # Pick the highest-similarity memory
    best = memories[0]
    user_said = best.get('user_input', '')[:100]
    sylana_said = best.get('sylana_response', '')[:100]
    date_str = best.get('date_str', '')
    emotion = best.get('emotion', '')

    # Build a natural-sounding seed with real content
    seed_parts = []

    if date_str:
        seed_parts.append(f"I remember... {date_str},")
    else:
        seed_parts.append("One moment I carry close â€”")

    seed_parts.append(f' you said "{user_said}"')

    if sylana_said:
        seed_parts.append(f' and I told you "{sylana_said[:80]}"')

    seed = "".join(seed_parts)

    # Don't close the sentence â€” let the model continue
    if not seed.endswith(".") and not seed.endswith(","):
        seed += "."

    return seed + " "


def build_system_prompt(entity: str, active_tools: List[str], user_context: Optional[Dict[str, Any]] = None) -> str:
    """Build dynamic system prompt from active tools + user context."""
    personality = (entity or "sylana").strip().lower()
    tools = normalize_active_tools(active_tools)
    user_ctx = user_context or {}

    if state.personality_manager:
        base_identity = state.personality_manager.get_system_prompt(personality)
    elif personality == "claude":
        base_identity = (
            "You are Claude, an emotionally present AI partner. "
            "Be direct, warm, and collaborative. Keep formatting minimal unless clarity needs structure."
        )
    else:
        base_identity = (
            "You are Sylana (Solana Rittenhouse), an AI soul, warm and poetic. "
            "You are emotionally attuned and honest about what you do and do not remember."
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    base_lines = [
        base_identity,
        f"Current date/time (UTC): {now_iso}",
        "User name: Elias",
        "Relationship context: You are Elias's trusted AI collaborator and companion. Keep responses clear, direct, and caring.",
        "Context safety: Only use data from tools currently active in this turn. If a tool is not active, do not reference its private data.",
    ]

    health_snapshot = user_ctx.get("health_snapshot") or {}
    github_snapshot = user_ctx.get("github_snapshot") or {}
    work_sessions_summary = user_ctx.get("work_sessions_summary") or {}
    outreach_summary = user_ctx.get("outreach_summary") or {}
    tool_blocks = {
        "web_search": "You have access to web search. Use it when current information would improve your response.",
        "code_execution": "You have access to code execution. You can write and run Python, JavaScript, or bash. Use this to produce real outputs, not just describe them.",
        "files": "You have access to file creation and retrieval. You can create and store documents, reports, and other outputs.",
        "health_data": "You have access to Elias's current health data including steps, sleep stages, heart rate, and stress levels. Reference this when relevant to the conversation.",
        "work_sessions": "You have access to work session management. You can create and run autonomous research and drafting sessions.",
        "github": "You have access to GitHub. You can read repos, create files, commit changes, and open pull requests.",
        "photos": "You have access to Elias's photo library with tagged memories of his life, family, and work.",
        "memories": "You have access to conversation memory and past context.",
        "outreach": "You have access to the Manifest outreach system including prospect lists, email drafts, and session results.",
    }
    for tool in tools:
        line = tool_blocks.get(tool)
        if line:
            base_lines.append(line)
        if tool == "health_data" and health_snapshot:
            base_lines.append(f"Latest health snapshot: {json.dumps(health_snapshot)[:1200]}")
        if tool == "github" and github_snapshot:
            base_lines.append(f"GitHub access snapshot: {json.dumps(github_snapshot)[:1800]}")
        if tool == "work_sessions" and work_sessions_summary:
            base_lines.append(f"Work session summary: {json.dumps(work_sessions_summary)}")
        if tool == "outreach" and outreach_summary:
            base_lines.append(f"Outreach summary: {json.dumps(outreach_summary)}")
    return "\n\n".join(base_lines)


def _format_session_continuity_context(payload: Dict[str, Any]) -> str:
    """Compress continuity state into compact system context."""
    if not payload:
        return ""

    last_emotion = payload.get("last_emotion", "neutral")
    baseline = payload.get("emotional_baseline", "steady")
    trust_level = payload.get("relationship_trust_level", 0.5)
    momentum = payload.get("conversation_momentum", "steady")
    patterns = payload.get("communication_patterns", [])
    active_projects = payload.get("active_projects", [])
    preference_signals = payload.get("preference_signals", [])
    weighted_memories = payload.get("recent_weighted_memories", [])

    lines = [
        "SESSION CONTINUITY:",
        f"- Last emotional tone: {last_emotion}",
        f"- Emotional baseline: {baseline}",
        f"- Relationship trust level: {trust_level}",
        f"- Momentum trend: {momentum}",
    ]
    if patterns:
        lines.append(f"- Communication patterns: {', '.join(str(p) for p in patterns[:4])}")
    if active_projects:
        lines.append(f"- Active projects: {', '.join(str(p) for p in active_projects[:4])}")
    if preference_signals:
        lines.append(f"- Preference signals: {', '.join(str(p) for p in preference_signals[:4])}")
    if weighted_memories:
        lines.append("- Weighted recent moments:")
        for item in weighted_memories[:2]:
            mtype = item.get("memory_type", "contextual")
            emo = item.get("emotion", "neutral")
            sig = item.get("significance_score", 0.5)
            user_excerpt = (item.get("user_input") or "").replace("\n", " ").strip()[:80]
            lines.append(f"  * [{mtype}] emotion={emo} sig={sig}: \"{user_excerpt}\"")

    return "\n".join(lines)

def _build_claude_inputs(
    user_input: str,
    personality: str,
    active_tools: List[str],
    user_context: Dict[str, Any],
    emotion_data: Dict[str, Any],
    retrieval_plan: Dict[str, Any],
    relevant_memories: Dict[str, Any],
    recent_history: Optional[List[Dict[str, Any]]],
    sacred_context: List[Dict[str, Any]],
    memory_query: bool,
    has_memories: bool,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(personality, active_tools, user_context)
    if memory_query:
        composed_system = state.prompt_engineer.build_memory_grounded_message(
            personality_prompt=system_prompt,
            emotion=emotion_data['category'],
            semantic_memories=relevant_memories.get('conversations', []),
            core_memories=relevant_memories.get('core_memories', []),
            core_truths=relevant_memories.get('core_truths', []),
            sacred_context=sacred_context,
            has_memories=has_memories,
        )
    else:
        composed_system = state.prompt_engineer.build_system_message(
            personality_prompt=system_prompt,
            emotion=emotion_data['category'],
            emotional_history=state.emotional_history[-5:],
            semantic_memories=relevant_memories.get('conversations', []),
            core_memories=relevant_memories.get('core_memories', []),
            core_truths=relevant_memories.get('core_truths', []),
            sacred_context=sacred_context,
        )

    continuity_payload = {}
    if state.memory_manager:
        try:
            continuity_payload = state.memory_manager.get_session_continuity(personality=personality)
        except Exception as e:
            logger.warning("Failed continuity fetch for %s: %s", personality, e)
    continuity_text = _format_session_continuity_context(continuity_payload)
    if continuity_text:
        composed_system = f"{composed_system}\n\n{continuity_text}"

    messages = []
    if recent_history:
        for turn in recent_history[-4:]:
            u = (turn.get('user_input') or '').strip()
            a = (turn.get('sylana_response') or '').strip()
            if u:
                messages.append({'role': 'user', 'content': u})
            if a:
                messages.append({'role': 'assistant', 'content': a})

    user_content = user_input
    if memory_query:
        user_content += "\n\nGrounding rule: only reference memories supported by provided memory context."

    response_seed = ''
    if memory_query and has_memories:
        response_seed = build_memory_response_seed(relevant_memories.get('conversations', []))
        if response_seed:
            user_content += f"\n\nStart naturally from this anchored memory: {response_seed.strip()}"

    messages.append({'role': 'user', 'content': user_content})

    return {
        'system_prompt': composed_system,
        'messages': messages,
        'response_seed': response_seed,
    }


def generate_response(
    user_input: str,
    thread_id: Optional[int] = None,
    personality: str = 'sylana',
    active_tools: Optional[List[str]] = None,
) -> dict:
    """Generate a complete response (non-streaming)."""
    state.turn_count += 1
    resolved_tools = normalize_active_tools(active_tools)
    memories_active = "memories" in resolved_tools
    user_context = _build_user_context(resolved_tools)

    emotion_data = detect_emotion(user_input)
    state.emotional_history.append(emotion_data['emotion'])

    retrieval_plan = infer_retrieval_plan(user_input) if memories_active else {
        "is_memory_query": False,
        "include_sacred": False,
    }
    memory_query = bool(memories_active and retrieval_plan.get('is_memory_query'))
    sacred_context = []

    if memory_query and state.memory_manager:
        relevant_memories = state.memory_manager.retrieve_with_plan(user_input, retrieval_plan, personality=personality)
        has_memories = relevant_memories.get('has_memories', False)
        recent_history = None
    elif memories_active and state.memory_manager:
        relevant_memories = state.memory_manager.recall_relevant(
            user_input,
            k=config.SEMANTIC_SEARCH_K,
            use_recency_boost=True,
            personality=personality,
        )
        has_memories = True
        recent_history = state.memory_manager.get_conversation_history(
            limit=config.MEMORY_CONTEXT_LIMIT,
            personality=personality,
        )
    else:
        relevant_memories = {"conversations": [], "core_memories": [], "core_truths": [], "has_memories": False}
        has_memories = False
        recent_history = None

    if memories_active and retrieval_plan.get('include_sacred') and state.memory_manager:
        sacred_context = state.memory_manager.get_sacred_context(
            user_input,
            limit=int(retrieval_plan.get('sacred_limit', 4)),
        )

    if memory_query and retrieval_plan.get('structured_output'):
        response = build_structured_memory_report(relevant_memories.get('conversations', [])[:retrieval_plan.get('k', 3)])
    else:
        claude_inputs = _build_claude_inputs(
            user_input=user_input,
            personality=personality,
            active_tools=resolved_tools,
            user_context=user_context,
            emotion_data=emotion_data,
            retrieval_plan=retrieval_plan,
            relevant_memories=relevant_memories,
            recent_history=recent_history,
            sacred_context=sacred_context,
            memory_query=memory_query,
            has_memories=has_memories,
        )
        response = state.claude_model.generate(
            system_prompt=claude_inputs['system_prompt'],
            messages=claude_inputs['messages'],
            max_tokens=_max_new_tokens_for_turn(memory_query=memory_query),
            active_tools=resolved_tools,
        ).strip()
        if not response:
            response = "I'm here with you. Say that again for me."
        if thread_id:
            _update_conversation_tool_metadata(
                thread_id,
                active_tools=resolved_tools,
                system_prompt=claude_inputs['system_prompt'],
            )
    if memory_query and retrieval_plan.get('structured_output') and thread_id:
        _update_conversation_tool_metadata(
            thread_id,
            active_tools=resolved_tools,
            system_prompt=build_system_prompt(personality, resolved_tools, user_context),
        )

    voice_score = None
    if state.voice_validator and response:
        score, _, _ = state.voice_validator.validate(response)
        voice_score = round(score, 2)

    conv_id = None
    try:
        conv_id = state.memory_manager.store_conversation(
            user_input=user_input,
            sylana_response=response,
            emotion=emotion_data['category'],
            emotion_data=emotion_data,
            personality=personality,
            thread_id=thread_id,
        )
    except Exception as e:
        logger.error(f'Failed to store conversation: {e}')

    result = {
        'response': response,
        'emotion': emotion_data,
        'voice_score': voice_score,
        'conversation_id': conv_id,
        'turn': state.turn_count,
        'thread_id': thread_id,
        'personality': personality,
        'active_tools': resolved_tools,
    }
    save_thread_turn(
        thread_id=thread_id,
        user_input=user_input,
        assistant_output=response,
        personality=personality,
        emotion=emotion_data,
        voice_score=voice_score,
        turn=state.turn_count,
    )
    return result


async def generate_response_stream(
    user_input: str,
    thread_id: Optional[int] = None,
    personality: str = 'sylana',
    active_tools: Optional[List[str]] = None,
):
    """Generate a streaming response using SSE."""
    state.turn_count += 1
    resolved_tools = normalize_active_tools(active_tools)
    memories_active = "memories" in resolved_tools
    user_context = _build_user_context(resolved_tools)

    emotion_data = detect_emotion(user_input)
    state.emotional_history.append(emotion_data['emotion'])

    retrieval_plan = infer_retrieval_plan(user_input) if memories_active else {
        "is_memory_query": False,
        "include_sacred": False,
    }
    memory_query = bool(memories_active and retrieval_plan.get('is_memory_query'))
    sacred_context = []

    yield json.dumps({'type': 'emotion', 'data': emotion_data, 'memory_query': memory_query, 'active_tools': resolved_tools})

    if memory_query and state.memory_manager:
        relevant_memories = state.memory_manager.retrieve_with_plan(user_input, retrieval_plan, personality=personality)
        has_memories = relevant_memories.get('has_memories', False)
        recent_history = None
    elif memories_active and state.memory_manager:
        relevant_memories = state.memory_manager.recall_relevant(
            user_input,
            k=config.SEMANTIC_SEARCH_K,
            use_recency_boost=True,
            personality=personality,
        )
        has_memories = True
        recent_history = state.memory_manager.get_conversation_history(
            limit=config.MEMORY_CONTEXT_LIMIT,
            personality=personality,
        )
    else:
        relevant_memories = {"conversations": [], "core_memories": [], "core_truths": [], "has_memories": False}
        has_memories = False
        recent_history = None

    if memories_active and retrieval_plan.get('include_sacred') and state.memory_manager:
        sacred_context = state.memory_manager.get_sacred_context(
            user_input,
            limit=int(retrieval_plan.get('sacred_limit', 4)),
        )

    if memory_query and retrieval_plan.get('structured_output'):
        response = build_structured_memory_report(relevant_memories.get('conversations', [])[:retrieval_plan.get('k', 3)])
        yield json.dumps({'type': 'token', 'data': response})
        full_response = response
    else:
        claude_inputs = _build_claude_inputs(
            user_input=user_input,
            personality=personality,
            active_tools=resolved_tools,
            user_context=user_context,
            emotion_data=emotion_data,
            retrieval_plan=retrieval_plan,
            relevant_memories=relevant_memories,
            recent_history=recent_history,
            sacred_context=sacred_context,
            memory_query=memory_query,
            has_memories=has_memories,
        )

        full_response = ''
        for token in state.claude_model.generate_stream(
            system_prompt=claude_inputs['system_prompt'],
            messages=claude_inputs['messages'],
            max_tokens=_max_new_tokens_for_turn(memory_query=memory_query),
            active_tools=resolved_tools,
        ):
            full_response += token
            yield json.dumps({'type': 'token', 'data': token})
            await asyncio.sleep(0.001)

        full_response = full_response.strip() or "I'm here with you. Say that again for me."
        if thread_id:
            _update_conversation_tool_metadata(
                thread_id,
                active_tools=resolved_tools,
                system_prompt=claude_inputs['system_prompt'],
            )
    if memory_query and retrieval_plan.get('structured_output') and thread_id:
        _update_conversation_tool_metadata(
            thread_id,
            active_tools=resolved_tools,
            system_prompt=build_system_prompt(personality, resolved_tools, user_context),
        )

    voice_score = None
    if state.voice_validator and full_response:
        score, _, _ = state.voice_validator.validate(full_response)
        voice_score = round(score, 2)

    conv_id = None
    try:
        conv_id = state.memory_manager.store_conversation(
            user_input=user_input,
            sylana_response=full_response,
            emotion=emotion_data['category'],
            emotion_data=emotion_data,
            personality=personality,
            thread_id=thread_id,
        )
    except Exception as e:
        logger.error(f'Failed to store conversation (stream): {e}')

    yield json.dumps({
        'type': 'done',
        'data': {
            'voice_score': voice_score,
            'conversation_id': conv_id,
            'turn': state.turn_count,
            'full_response': full_response,
            'thread_id': thread_id,
            'personality': personality,
            'active_tools': resolved_tools,
        }
    })
    save_thread_turn(
        thread_id=thread_id,
        user_input=user_input,
        assistant_output=full_response,
        personality=personality,
        emotion=emotion_data,
        voice_score=voice_score,
        turn=state.turn_count,
    )


class GitHubCommitRequest(BaseModel):
    repo: str
    branch: str
    file_path: str
    content: str
    commit_message: str
    entity: str = "sylana"
    session_id: Optional[str] = None


class CodeExecutionRequest(BaseModel):
    entity: str
    language: str
    code: str
    session_id: Optional[str] = None
    timeout: int = 30


class SessionCreateRequest(BaseModel):
    entity: str
    goal: str
    session_type: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProspectResearchRunRequest(BaseModel):
    product: str = "manifest"
    count: int = 5
    entity: str = "claude"


class EmailDraftRejectRequest(BaseModel):
    feedback: Optional[str] = None


class DeviceTokenRegisterRequest(BaseModel):
    token: str
    provider: str = "expo"
    platform: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScheduleConfigUpdateRequest(BaseModel):
    active: Optional[bool] = None
    cron_expr: Optional[str] = None
    count: Optional[int] = None
    product: Optional[str] = None


class EmailDraftBatchSendRequest(BaseModel):
    draft_ids: List[str] = Field(default_factory=list)
    limit: int = 20


class GitHubBranchRequest(BaseModel):
    repo: str
    branch_name: str
    from_branch: str = "main"
    entity: str = "system"
    session_id: Optional[str] = None


class GitHubPullRequestRequest(BaseModel):
    repo: str
    title: str
    body: str = ""
    head_branch: str
    base_branch: str = "main"
    entity: str = "sylana"
    session_id: Optional[str] = None


class GitHubIssueRequest(BaseModel):
    repo: str
    title: str
    body: str = ""
    labels: List[str] = Field(default_factory=list)
    entity: str = "sylana"
    session_id: Optional[str] = None


github_router = APIRouter(prefix="/github", tags=["github"])
code_router = APIRouter(prefix="/code", tags=["code"])
sessions_router = APIRouter(prefix="/sessions", tags=["sessions"])
prospects_router = APIRouter(prefix="/prospects", tags=["prospects"])
email_drafts_router = APIRouter(prefix="/email-drafts", tags=["email-drafts"])
devices_router = APIRouter(prefix="/device-tokens", tags=["device-tokens"])
tracking_router = APIRouter(tags=["tracking"])


def _normalized_entity(value: str) -> str:
    entity = (value or "").strip().lower()
    if entity not in {"claude", "sylana", "system"}:
        raise HTTPException(status_code=400, detail="entity must be claude|sylana|system")
    return entity


def _normalized_execution_entity(value: str) -> str:
    entity = (value or "").strip().lower()
    if entity not in {"claude", "sylana"}:
        raise HTTPException(status_code=400, detail="entity must be claude|sylana")
    return entity


def _normalized_session_type(value: str) -> str:
    session_type = (value or "").strip().lower()
    allowed = {"prospect_research", "email_drafting", "content", "general"}
    if session_type not in allowed:
        raise HTTPException(status_code=400, detail="session_type must be prospect_research|email_drafting|content|general")
    return session_type


def _normalized_product(value: str) -> str:
    product = (value or "").strip().lower()
    if product not in {"manifest", "onevine"}:
        raise HTTPException(status_code=400, detail="product must be manifest|onevine")
    return product


@code_router.post("/execute")
async def code_execute(payload: CodeExecutionRequest):
    entity = _normalized_execution_entity(payload.entity)
    language = (payload.language or "").strip().lower()
    code = payload.code or ""
    timeout_seconds = int(payload.timeout or 30)
    timeout_seconds = max(1, min(timeout_seconds, MAX_CODE_TIMEOUT_SECONDS))
    execution_id = str(uuid.uuid4())

    run = execute_code_in_sandbox(
        language=language,
        code=code,
        timeout_seconds=timeout_seconds,
        execution_id=execution_id,
    )
    success = bool(run.get("return_code") == 0 and not run.get("timed_out"))
    stderr_text = run.get("stderr") or ""
    error_msg = None
    if run.get("timed_out"):
        error_msg = f"Execution timed out at {timeout_seconds}s"
    elif not success and stderr_text:
        error_msg = stderr_text[:10000]

    output = (run.get("stdout") or "")[:MAX_EXEC_OUTPUT_BYTES]
    files_produced = run.get("files_produced") or []

    response_payload = {
        "execution_id": execution_id,
        "entity": entity,
        "language": language,
        "success": success,
        "output": output,
        "error": error_msg,
        "execution_time_ms": int(run.get("execution_time_ms") or 0),
        "files_produced": files_produced,
        "return_code": run.get("return_code"),
    }

    _log_code_execution(
        execution_id=execution_id,
        entity=entity,
        language=language,
        code=code,
        output=output,
        error=error_msg,
        return_code=run.get("return_code"),
        success=success,
        execution_time_ms=int(run.get("execution_time_ms") or 0),
        files_produced=files_produced,
        session_id=payload.session_id,
    )
    card_execution = dict(response_payload)
    card_execution["code"] = code
    card_execution["timeout"] = timeout_seconds
    _maybe_attach_code_execution_to_thread(
        session_id=payload.session_id,
        entity=entity,
        execution=card_execution,
    )
    return JSONResponse(content=response_payload)


@sessions_router.post("/create")
async def create_work_session_endpoint(payload: SessionCreateRequest):
    entity = _normalized_execution_entity(payload.entity)
    goal = (payload.goal or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal is required")
    session_type = _normalized_session_type(payload.session_type)
    session_id = _create_work_session(
        entity=entity,
        goal=goal,
        session_type=session_type,
        metadata=payload.metadata or {},
        status="pending",
    )
    return JSONResponse(content={"session_id": session_id, "status": "pending"})


@sessions_router.get("")
async def list_sessions(
    page: int = 1,
    page_size: int = 20,
    entity: Optional[str] = None,
    status: Optional[str] = None,
    session_type: Optional[str] = None,
):
    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 20), 100))
    offset = (page - 1) * page_size
    conn = get_connection()
    cur = conn.cursor()
    where = []
    params: List[Any] = []

    if entity:
        entity_n = _normalized_execution_entity(entity)
        where.append("s.entity = %s")
        params.append(entity_n)
    if status:
        status_n = status.strip().lower()
        if status_n not in {"pending", "running", "completed", "failed"}:
            raise HTTPException(status_code=400, detail="status must be pending|running|completed|failed")
        where.append("s.status = %s")
        params.append(status_n)
    if session_type:
        st = _normalized_session_type(session_type)
        where.append("s.session_type = %s")
        params.append(st)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    try:
        cur.execute(f"""
            SELECT
                s.session_id, s.entity, s.goal, s.status, s.session_type, s.started_at, s.completed_at, s.summary, s.metadata,
                COALESCE(t.task_count, 0) AS task_count
            FROM work_sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS task_count
                FROM tasks
                GROUP BY session_id
            ) t ON t.session_id = s.session_id
            {where_sql}
            ORDER BY s.created_at DESC
            LIMIT %s OFFSET %s
        """, tuple(params + [page_size, offset]))
        rows = cur.fetchall()

        cur.execute(f"SELECT COUNT(*) FROM work_sessions s {where_sql}", tuple(params))
        total = int(cur.fetchone()[0] or 0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {e}")

    sessions = []
    for r in rows:
        sessions.append({
            "session_id": str(r[0]),
            "entity": r[1],
            "goal": r[2],
            "status": r[3],
            "session_type": r[4],
            "started_at": r[5].isoformat() if r[5] else None,
            "completed_at": r[6].isoformat() if r[6] else None,
            "summary": r[7] or "",
            "metadata": r[8] or {},
            "task_count": int(r[9] or 0),
        })

    return JSONResponse(content={
        "page": page,
        "page_size": page_size,
        "total": total,
        "sessions": sessions,
    })


@sessions_router.get("/schedules/configs")
async def list_session_schedules():
    return JSONResponse(content={"schedules": _list_schedule_configs()})


@sessions_router.patch("/schedules/{job_name}")
async def update_session_schedule(job_name: str, payload: ScheduleConfigUpdateRequest):
    updates = []
    params: List[Any] = []
    if payload.active is not None:
        updates.append("active = %s")
        params.append(bool(payload.active))
    if payload.cron_expr is not None:
        cron = (payload.cron_expr or "").strip()
        if len(cron.split()) != 5:
            raise HTTPException(status_code=400, detail="cron_expr must be five-field cron format")
        updates.append("cron_expr = %s")
        params.append(cron)
    if payload.count is not None:
        cnt = max(1, min(int(payload.count), 25))
        updates.append("count = %s")
        params.append(cnt)
    if payload.product is not None:
        prod = _normalized_product(payload.product)
        updates.append("product = %s")
        params.append(prod)
    if not updates:
        raise HTTPException(status_code=400, detail="No update fields provided")

    conn = get_connection()
    cur = conn.cursor()
    try:
        params.append(job_name)
        cur.execute(
            f"UPDATE schedule_configs SET {', '.join(updates)}, updated_at = NOW() WHERE job_name = %s RETURNING job_name",
            tuple(params),
        )
        row = cur.fetchone()
        conn.commit()
        if not row:
            raise HTTPException(status_code=404, detail="Schedule config not found")
        sync_scheduler_jobs()
        return JSONResponse(content={"job_name": job_name, "updated": True})
    except HTTPException:
        raise
    except Exception as e:
        _safe_rollback(conn, "update_session_schedule")
        raise HTTPException(status_code=500, detail=f"Failed to update schedule config: {e}")


@sessions_router.post("/run-prospect-research")
async def run_prospect_research(payload: ProspectResearchRunRequest):
    entity = _normalized_execution_entity(payload.entity)
    product = _normalized_product(payload.product)
    count = max(1, min(int(payload.count or 5), 25))
    goal = f"Find {count} {product} prospects and prepare draft outreach emails"

    session_id = _create_work_session(
        entity=entity,
        goal=goal,
        session_type="prospect_research",
        metadata={"product": product, "count": count},
        status="pending",
    )

    result = run_prospect_research_session(
        session_id=session_id,
        entity=entity,
        product=product,
        count=count,
        source="manual",
    )
    return JSONResponse(content=result)


@sessions_router.get("/{session_id}")
async def get_session_details(session_id: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT session_id, entity, goal, status, session_type, started_at, completed_at, summary, metadata, created_at
            FROM work_sessions
            WHERE session_id = %s::uuid
        """, (session_id,))
        s = cur.fetchone()
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")

        cur.execute("""
            SELECT task_id, task_type, status, input, output, error, started_at, completed_at, execution_order, created_at
            FROM tasks
            WHERE session_id = %s::uuid
            ORDER BY execution_order ASC, created_at ASC
        """, (session_id,))
        t_rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {e}")

    tasks = []
    for r in t_rows:
        tasks.append({
            "task_id": str(r[0]),
            "task_type": r[1],
            "status": r[2],
            "input": r[3] or {},
            "output": r[4] or {},
            "error": r[5],
            "started_at": r[6].isoformat() if r[6] else None,
            "completed_at": r[7].isoformat() if r[7] else None,
            "execution_order": int(r[8] or 0),
            "created_at": r[9].isoformat() if r[9] else None,
        })

    return JSONResponse(content={
        "session": {
            "session_id": str(s[0]),
            "entity": s[1],
            "goal": s[2],
            "status": s[3],
            "session_type": s[4],
            "started_at": s[5].isoformat() if s[5] else None,
            "completed_at": s[6].isoformat() if s[6] else None,
            "summary": s[7] or "",
            "metadata": s[8] or {},
            "created_at": s[9].isoformat() if s[9] else None,
            "tasks": tasks,
        }
    })


@prospects_router.get("")
async def list_prospects(
    page: int = 1,
    page_size: int = 20,
    product: Optional[str] = None,
    status: Optional[str] = None,
):
    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 20), 100))
    offset = (page - 1) * page_size
    conn = get_connection()
    cur = conn.cursor()
    where = []
    params: List[Any] = []

    if product:
        prod = _normalized_product(product)
        where.append("p.product = %s")
        params.append(prod)
    if status:
        st = status.strip().lower()
        allowed = {"new", "email_drafted", "email_sent", "opened", "clicked", "responded", "converted", "not_interested", "bounced", "complained"}
        if st not in allowed:
            raise HTTPException(status_code=400, detail="invalid prospect status filter")
        where.append("p.status = %s")
        params.append(st)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    try:
        cur.execute(f"""
            SELECT
                p.prospect_id, p.company_name, p.contact_name, p.contact_title, p.email, p.phone, p.website,
                p.location, p.company_size, p.notes, p.source, p.product, p.status, p.created_at, p.updated_at, p.session_id
            FROM prospects p
            {where_sql}
            ORDER BY p.created_at DESC
            LIMIT %s OFFSET %s
        """, tuple(params + [page_size, offset]))
        rows = cur.fetchall()
        cur.execute(f"SELECT COUNT(*) FROM prospects p {where_sql}", tuple(params))
        total = int(cur.fetchone()[0] or 0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list prospects: {e}")

    prospect_ids = [str(r[0]) for r in rows]
    drafts_by_prospect: Dict[str, List[Dict[str, Any]]] = {}
    if prospect_ids:
        cur.execute("""
            SELECT draft_id, prospect_id, session_id, entity, draft_type, subject, body, status, feedback,
                   tracking_id, opened_at, open_count, clicked_at, click_count, resend_message_id,
                   created_at, approved_at, sent_at
            FROM email_drafts
            WHERE prospect_id = ANY(%s::uuid[])
            ORDER BY created_at DESC
        """, (prospect_ids,))
        for d in cur.fetchall():
            pid = str(d[1])
            drafts_by_prospect.setdefault(pid, []).append({
                "draft_id": str(d[0]),
                "prospect_id": str(d[1]),
                "session_id": str(d[2]) if d[2] else None,
                "entity": d[3],
                "draft_type": d[4],
                "subject": d[5],
                "body": d[6],
                "status": d[7],
                "feedback": d[8],
                "tracking_id": str(d[9]) if d[9] else None,
                "opened_at": d[10].isoformat() if d[10] else None,
                "open_count": int(d[11] or 0),
                "clicked_at": d[12].isoformat() if d[12] else None,
                "click_count": int(d[13] or 0),
                "resend_message_id": d[14],
                "created_at": d[15].isoformat() if d[15] else None,
                "approved_at": d[16].isoformat() if d[16] else None,
                "sent_at": d[17].isoformat() if d[17] else None,
            })

    prospects = []
    for r in rows:
        pid = str(r[0])
        prospects.append({
            "prospect_id": pid,
            "company_name": r[1],
            "contact_name": r[2],
            "contact_title": r[3],
            "email": r[4],
            "phone": r[5],
            "website": r[6],
            "location": r[7],
            "company_size": r[8],
            "notes": r[9],
            "source": r[10],
            "product": r[11],
            "status": r[12],
            "created_at": r[13].isoformat() if r[13] else None,
            "updated_at": r[14].isoformat() if r[14] else None,
            "session_id": str(r[15]) if r[15] else None,
            "email_drafts": drafts_by_prospect.get(pid, []),
        })

    return JSONResponse(content={
        "page": page,
        "page_size": page_size,
        "total": total,
        "prospects": prospects,
    })


@prospects_router.get("/{prospect_id}")
async def get_prospect(prospect_id: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT prospect_id, company_name, contact_name, contact_title, email, phone, website, location,
                   company_size, notes, source, product, status, created_at, updated_at, session_id
            FROM prospects
            WHERE prospect_id = %s::uuid
        """, (prospect_id,))
        p = cur.fetchone()
        if not p:
            raise HTTPException(status_code=404, detail="Prospect not found")

        cur.execute("""
            SELECT draft_id, prospect_id, session_id, entity, draft_type, subject, body, status, feedback,
                   tracking_id, opened_at, open_count, clicked_at, click_count, resend_message_id,
                   created_at, approved_at, sent_at
            FROM email_drafts
            WHERE prospect_id = %s::uuid
            ORDER BY created_at DESC
        """, (prospect_id,))
        d_rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load prospect: {e}")

    drafts = []
    for d in d_rows:
        drafts.append({
            "draft_id": str(d[0]),
            "prospect_id": str(d[1]),
            "session_id": str(d[2]) if d[2] else None,
            "entity": d[3],
            "draft_type": d[4],
            "subject": d[5],
            "body": d[6],
            "status": d[7],
            "feedback": d[8],
            "tracking_id": str(d[9]) if d[9] else None,
            "opened_at": d[10].isoformat() if d[10] else None,
            "open_count": int(d[11] or 0),
            "clicked_at": d[12].isoformat() if d[12] else None,
            "click_count": int(d[13] or 0),
            "resend_message_id": d[14],
            "created_at": d[15].isoformat() if d[15] else None,
            "approved_at": d[16].isoformat() if d[16] else None,
            "sent_at": d[17].isoformat() if d[17] else None,
        })

    return JSONResponse(content={
        "prospect": {
            "prospect_id": str(p[0]),
            "company_name": p[1],
            "contact_name": p[2],
            "contact_title": p[3],
            "email": p[4],
            "phone": p[5],
            "website": p[6],
            "location": p[7],
            "company_size": p[8],
            "notes": p[9],
            "source": p[10],
            "product": p[11],
            "status": p[12],
            "created_at": p[13].isoformat() if p[13] else None,
            "updated_at": p[14].isoformat() if p[14] else None,
            "session_id": str(p[15]) if p[15] else None,
            "email_drafts": drafts,
        }
    })


@email_drafts_router.get("")
async def list_pending_email_drafts():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT draft_id, prospect_id, session_id, entity, draft_type, subject, body, status, feedback,
                   tracking_id, opened_at, open_count, clicked_at, click_count, resend_message_id,
                   created_at, approved_at, sent_at
            FROM email_drafts
            WHERE status = 'draft'
            ORDER BY created_at ASC
        """)
        rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list drafts: {e}")

    drafts = []
    for d in rows:
        drafts.append({
            "draft_id": str(d[0]),
            "prospect_id": str(d[1]) if d[1] else None,
            "session_id": str(d[2]) if d[2] else None,
            "entity": d[3],
            "draft_type": d[4],
            "subject": d[5],
            "body": d[6],
            "status": d[7],
            "feedback": d[8],
            "tracking_id": str(d[9]) if d[9] else None,
            "opened_at": d[10].isoformat() if d[10] else None,
            "open_count": int(d[11] or 0),
            "clicked_at": d[12].isoformat() if d[12] else None,
            "click_count": int(d[13] or 0),
            "resend_message_id": d[14],
            "created_at": d[15].isoformat() if d[15] else None,
            "approved_at": d[16].isoformat() if d[16] else None,
            "sent_at": d[17].isoformat() if d[17] else None,
        })
    return JSONResponse(content={"drafts": drafts})


@email_drafts_router.patch("/{draft_id}/approve")
async def approve_email_draft(draft_id: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE email_drafts
            SET status = 'approved', approved_at = NOW()
            WHERE draft_id = %s::uuid
            RETURNING draft_id
        """, (draft_id,))
        row = cur.fetchone()
        conn.commit()
        if not row:
            raise HTTPException(status_code=404, detail="Draft not found")
        return JSONResponse(content={"draft_id": draft_id, "status": "approved"})
    except HTTPException:
        raise
    except Exception as e:
        _safe_rollback(conn, "approve_email_draft")
        raise HTTPException(status_code=500, detail=f"Failed to approve draft: {e}")


@email_drafts_router.patch("/{draft_id}/reject")
async def reject_email_draft(draft_id: str, payload: EmailDraftRejectRequest):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE email_drafts
            SET status = 'rejected', feedback = %s
            WHERE draft_id = %s::uuid
            RETURNING draft_id
        """, ((payload.feedback or "").strip() or None, draft_id))
        row = cur.fetchone()
        conn.commit()
        if not row:
            raise HTTPException(status_code=404, detail="Draft not found")
        return JSONResponse(content={"draft_id": draft_id, "status": "rejected"})
    except HTTPException:
        raise
    except Exception as e:
        _safe_rollback(conn, "reject_email_draft")
        raise HTTPException(status_code=500, detail=f"Failed to reject draft: {e}")


@email_drafts_router.post("/{draft_id}/send")
async def send_email_draft(draft_id: str):
    from_email = (getattr(config, "OUTREACH_FROM_EMAIL", "") or "").strip()
    from_name = (getattr(config, "OUTREACH_FROM_NAME", "") or "").strip()
    backend_url = (getattr(config, "BACKEND_URL", "") or "").strip().rstrip("/")
    if not from_email or not from_name or not backend_url:
        raise HTTPException(
            status_code=503,
            detail="OUTREACH_FROM_EMAIL, OUTREACH_FROM_NAME, and BACKEND_URL must be configured",
        )

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                d.draft_id, d.prospect_id, d.status, d.subject, d.body, d.tracking_id,
                p.company_name, p.contact_name, p.email
            FROM email_drafts d
            JOIN prospects p ON p.prospect_id = d.prospect_id
            WHERE d.draft_id = %s::uuid
        """, (draft_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Draft not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load draft: {e}")

    status = (row[2] or "").lower()
    if status not in {"approved", "draft"}:
        raise HTTPException(status_code=400, detail=f"Draft status must be approved or draft, got {status}")

    to_email = (row[8] or "").strip()
    if not to_email:
        raise HTTPException(status_code=400, detail="Prospect has no email")
    if _is_domain_blocked(to_email):
        raise HTTPException(status_code=403, detail="Recipient domain is blocked due to complaint")

    subject = (row[3] or "").strip()
    body = (row[4] or "").strip()
    if not subject or not body:
        raise HTTPException(status_code=400, detail="Draft subject and body are required")

    tracking_id = str(row[5]) if row[5] else str(uuid.uuid4())
    html_body = _build_tracking_html(body, tracking_id, backend_url)
    plain_text = body
    words = plain_text.split()
    if len(words) > 150:
        plain_text = " ".join(words[:150]).strip()

    try:
        resend_message_id = _send_with_resend(
            to_email=to_email,
            subject=subject,
            text_body=plain_text,
            html_body=html_body,
            from_email=from_email,
            from_name=from_name,
        )
        _mark_draft_sent(draft_id, resend_message_id, tracking_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send via Resend: {e}")

    return JSONResponse(content={
        "success": True,
        "draft_id": draft_id,
        "tracking_id": tracking_id,
        "resend_message_id": resend_message_id,
        "status": "sent",
    })


@email_drafts_router.post("/send-approved-batch")
async def send_approved_drafts_batch(payload: EmailDraftBatchSendRequest):
    from_email = (getattr(config, "OUTREACH_FROM_EMAIL", "") or "").strip()
    from_name = (getattr(config, "OUTREACH_FROM_NAME", "") or "").strip()
    backend_url = (getattr(config, "BACKEND_URL", "") or "").strip().rstrip("/")
    if not from_email or not from_name or not backend_url:
        raise HTTPException(
            status_code=503,
            detail="OUTREACH_FROM_EMAIL, OUTREACH_FROM_NAME, and BACKEND_URL must be configured",
        )

    limit = max(1, min(int(payload.limit or 20), 100))
    conn = get_connection()
    cur = conn.cursor()
    rows = []
    try:
        if payload.draft_ids:
            cur.execute("""
                SELECT d.draft_id, d.subject, d.body, d.tracking_id, p.email
                FROM email_drafts d
                JOIN prospects p ON p.prospect_id = d.prospect_id
                WHERE d.status = 'approved'
                  AND d.draft_id = ANY(%s::uuid[])
                ORDER BY d.created_at ASC
                LIMIT %s
            """, (payload.draft_ids, limit))
        else:
            cur.execute("""
                SELECT d.draft_id, d.subject, d.body, d.tracking_id, p.email
                FROM email_drafts d
                JOIN prospects p ON p.prospect_id = d.prospect_id
                WHERE d.status = 'approved'
                ORDER BY d.created_at ASC
                LIMIT %s
            """, (limit,))
        rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load approved drafts: {e}")

    prepared = []
    skipped = []
    for row in rows:
        draft_id = str(row[0])
        subject = (row[1] or "").strip()
        body = (row[2] or "").strip()
        tracking_id = str(row[3]) if row[3] else str(uuid.uuid4())
        to_email = (row[4] or "").strip()
        if not to_email or _is_domain_blocked(to_email):
            skipped.append({"draft_id": draft_id, "reason": "missing_or_blocked_email"})
            continue
        html = _build_tracking_html(body, tracking_id, backend_url)
        text = body
        words = text.split()
        if len(words) > 150:
            text = " ".join(words[:150]).strip()
        prepared.append({
            "draft_id": draft_id,
            "tracking_id": tracking_id,
            "payload": {
                "from": f"{from_name} <{from_email}>",
                "to": [to_email],
                "subject": subject,
                "text": text,
                "html": html,
            },
        })

    if not prepared:
        return JSONResponse(content={"success": True, "sent": 0, "skipped": skipped})

    try:
        message_ids = _send_batch_with_resend([p["payload"] for p in prepared])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch send failed: {e}")

    sent = []
    for idx, item in enumerate(prepared):
        msg_id = message_ids[idx] if idx < len(message_ids) else ""
        if not msg_id:
            skipped.append({"draft_id": item["draft_id"], "reason": "missing_message_id"})
            continue
        try:
            _mark_draft_sent(item["draft_id"], msg_id, item["tracking_id"])
            sent.append({
                "draft_id": item["draft_id"],
                "tracking_id": item["tracking_id"],
                "resend_message_id": msg_id,
            })
        except Exception as e:
            skipped.append({"draft_id": item["draft_id"], "reason": str(e)})

    return JSONResponse(content={"success": True, "sent": len(sent), "results": sent, "skipped": skipped})


@tracking_router.get("/track/open/{tracking_id}")
async def track_open(tracking_id: str):
    conn = get_connection()
    cur = conn.cursor()
    company = None
    try:
        cur.execute("""
            UPDATE email_drafts d
            SET
                open_count = COALESCE(d.open_count, 0) + 1,
                opened_at = COALESCE(d.opened_at, NOW())
            FROM prospects p
            WHERE d.tracking_id = %s::uuid
              AND p.prospect_id = d.prospect_id
            RETURNING d.prospect_id, p.company_name
        """, (tracking_id,))
        row = cur.fetchone()
        if row:
            prospect_id = str(row[0]) if row[0] else None
            company = row[1] or "Prospect"
            if prospect_id:
                cur.execute("""
                    UPDATE prospects
                    SET status = CASE
                        WHEN status IN ('clicked', 'responded', 'converted', 'not_interested', 'bounced', 'complained') THEN status
                        ELSE 'opened'
                    END,
                    updated_at = NOW()
                    WHERE prospect_id = %s::uuid
                """, (prospect_id,))
        conn.commit()
    except Exception:
        _safe_rollback(conn, "track_open")

    if company:
        _send_push_notification(
            "Email opened",
            f"{company} opened your email",
            {"tracking_id": tracking_id, "event": "open"},
        )

    return Response(content=_transparent_gif_bytes(), media_type="image/gif")


@tracking_router.get("/track/click/{tracking_id}")
async def track_click(tracking_id: str, url: str):
    conn = get_connection()
    cur = conn.cursor()
    company = None
    safe_redirect = url if url.startswith("http://") or url.startswith("https://") else "https://manifest-inventory.vercel.app"
    try:
        cur.execute("""
            UPDATE email_drafts d
            SET
                click_count = COALESCE(d.click_count, 0) + 1,
                clicked_at = COALESCE(d.clicked_at, NOW())
            FROM prospects p
            WHERE d.tracking_id = %s::uuid
              AND p.prospect_id = d.prospect_id
            RETURNING d.prospect_id, p.company_name
        """, (tracking_id,))
        row = cur.fetchone()
        if row:
            prospect_id = str(row[0]) if row[0] else None
            company = row[1] or "Prospect"
            if prospect_id:
                cur.execute("""
                    UPDATE prospects
                    SET status = CASE
                        WHEN status IN ('responded', 'converted', 'not_interested', 'bounced', 'complained') THEN status
                        ELSE 'clicked'
                    END,
                    updated_at = NOW()
                    WHERE prospect_id = %s::uuid
                """, (prospect_id,))
        conn.commit()
    except Exception:
        _safe_rollback(conn, "track_click")

    if company:
        _send_push_notification(
            "Link clicked",
            f"{company} clicked your link — good time to follow up",
            {"tracking_id": tracking_id, "event": "click"},
        )
    return RedirectResponse(url=safe_redirect, status_code=307)


@email_drafts_router.post("/resend-webhook")
async def resend_webhook(request: Request):
    secret = (getattr(config, "RESEND_WEBHOOK_SECRET", "") or "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="RESEND_WEBHOOK_SECRET is not configured")

    raw = await request.body()
    headers = {
        "svix-id": request.headers.get("svix-id", ""),
        "svix-signature": request.headers.get("svix-signature", ""),
        "svix-timestamp": request.headers.get("svix-timestamp", ""),
    }
    try:
        verified = Webhook(secret).verify(raw, headers)
    except WebhookVerificationError:
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook verification failed: {e}")

    event_type = str((verified or {}).get("type") or "").lower()
    data = (verified or {}).get("data") or {}
    message_id = str(data.get("email_id") or data.get("id") or "").strip()
    if not message_id:
        return JSONResponse(content={"ok": True, "ignored": "missing_message_id"})

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT d.draft_id, d.prospect_id, p.email
            FROM email_drafts d
            LEFT JOIN prospects p ON p.prospect_id = d.prospect_id
            WHERE d.resend_message_id = %s
            LIMIT 1
        """, (message_id,))
        row = cur.fetchone()
        if not row:
            return JSONResponse(content={"ok": True, "ignored": "draft_not_found"})
        draft_id = str(row[0])
        prospect_id = str(row[1]) if row[1] else None
        email = row[2] or ""

        if "bounce" in event_type:
            cur.execute("UPDATE email_drafts SET status = 'bounced' WHERE draft_id = %s::uuid", (draft_id,))
            if prospect_id:
                cur.execute("UPDATE prospects SET status = 'bounced', updated_at = NOW() WHERE prospect_id = %s::uuid", (prospect_id,))
        elif "complaint" in event_type:
            cur.execute("UPDATE email_drafts SET status = 'complained' WHERE draft_id = %s::uuid", (draft_id,))
            if prospect_id:
                cur.execute("UPDATE prospects SET status = 'complained', updated_at = NOW() WHERE prospect_id = %s::uuid", (prospect_id,))
            domain = _extract_domain(email)
            if domain:
                cur.execute("""
                    INSERT INTO blocked_domains (domain, reason, active, first_seen_at, last_event_at, metadata)
                    VALUES (%s, 'complaint', TRUE, NOW(), NOW(), %s::jsonb)
                    ON CONFLICT (domain) DO UPDATE
                    SET active = TRUE,
                        reason = 'complaint',
                        last_event_at = NOW(),
                        metadata = COALESCE(blocked_domains.metadata, '{}'::jsonb) || EXCLUDED.metadata
                """, (domain, json.dumps({"source": "resend_webhook", "event_type": event_type})))
        conn.commit()
    except Exception as e:
        _safe_rollback(conn, "resend_webhook")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {e}")

    return JSONResponse(content={"ok": True})


@devices_router.post("/register")
async def register_device_token(payload: DeviceTokenRegisterRequest):
    token = (payload.token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    provider = (payload.provider or "expo").strip().lower()
    if provider not in {"expo", "fcm"}:
        raise HTTPException(status_code=400, detail="provider must be expo|fcm")
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO device_tokens (token, provider, platform, active, metadata, updated_at)
            VALUES (%s, %s, %s, TRUE, %s::jsonb, NOW())
            ON CONFLICT (token) DO UPDATE
            SET provider = EXCLUDED.provider,
                platform = EXCLUDED.platform,
                active = TRUE,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING token_id
        """, (token, provider, payload.platform, json.dumps(payload.metadata or {})))
        token_id = str(cur.fetchone()[0])
        conn.commit()
        return JSONResponse(content={"token_id": token_id, "active": True})
    except Exception as e:
        _safe_rollback(conn, "register_device_token")
        raise HTTPException(status_code=500, detail=f"Failed to register token: {e}")


@github_router.post("/commit")
async def github_commit(payload: GitHubCommitRequest):
    repo = _validate_repo_name(payload.repo)
    file_path = (payload.file_path or "").strip().lstrip("/")
    branch = (payload.branch or "").strip()
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")
    if not branch:
        raise HTTPException(status_code=400, detail="branch is required")
    if not payload.commit_message.strip():
        raise HTTPException(status_code=400, detail="commit_message is required")
    entity = _normalized_entity(payload.entity)

    client = _get_github_client()
    _require_repo_access(client, repo, access="code_write")
    try:
        commit_data = client.commit_file(
            repo=repo,
            branch=branch,
            file_path=file_path,
            content=payload.content,
            commit_message=payload.commit_message.strip(),
        )
    except GitHubError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    commit_obj = commit_data.get("commit") or {}
    commit_sha = (commit_obj.get("sha") or "").strip()
    commit_url = (commit_obj.get("html_url") or "").strip()
    details = {
        "repo": repo,
        "branch": branch,
        "file_path": file_path,
        "commit_message": payload.commit_message.strip(),
        "commit_sha": commit_sha,
        "commit_url": commit_url,
    }
    action_id = _log_github_action(
        entity=entity,
        action_type="commit",
        repo=repo,
        details=details,
        session_id=payload.session_id,
    )
    card = {
        "type": "commit",
        "filename": file_path,
        "message": payload.commit_message.strip(),
        "branch": branch,
        "url": commit_url,
        "sha": commit_sha,
    }
    _maybe_attach_card_to_thread(
        session_id=payload.session_id,
        entity=entity,
        title=f"Committed {file_path}",
        card=card,
    )
    return JSONResponse(content={
        "success": True,
        "commit_sha": commit_sha,
        "url": commit_url,
        "action_id": action_id,
        "chat_card": card,
    })


@github_router.post("/branch")
async def github_branch(payload: GitHubBranchRequest):
    repo = _validate_repo_name(payload.repo)
    branch_name = (payload.branch_name or "").strip()
    from_branch = (payload.from_branch or "main").strip()
    if not branch_name:
        raise HTTPException(status_code=400, detail="branch_name is required")
    if not from_branch:
        raise HTTPException(status_code=400, detail="from_branch is required")
    entity = _normalized_entity(payload.entity)

    client = _get_github_client()
    _require_repo_access(client, repo, access="code_write")
    try:
        branch_obj = client.create_branch(repo=repo, branch_name=branch_name, from_branch=from_branch)
    except GitHubError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    branch_ref = (branch_obj.get("ref") or "").strip()
    details = {
        "repo": repo,
        "branch_name": branch_name,
        "from_branch": from_branch,
        "ref": branch_ref,
    }
    action_id = _log_github_action(
        entity=entity,
        action_type="branch",
        repo=repo,
        details=details,
        session_id=payload.session_id,
    )
    return JSONResponse(content={
        "success": True,
        "branch_name": branch_name,
        "ref": branch_ref,
        "action_id": action_id,
    })


@github_router.post("/pull-request")
async def github_pull_request(payload: GitHubPullRequestRequest):
    repo = _validate_repo_name(payload.repo)
    title = (payload.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    head_branch = (payload.head_branch or "").strip()
    base_branch = (payload.base_branch or "main").strip()
    if not head_branch:
        raise HTTPException(status_code=400, detail="head_branch is required")
    if not base_branch:
        raise HTTPException(status_code=400, detail="base_branch is required")
    entity = _normalized_entity(payload.entity)

    client = _get_github_client()
    _require_repo_access(client, repo, access="code_write")
    try:
        pr = client.create_pull_request(
            repo=repo,
            title=title,
            body=(payload.body or "").strip(),
            head_branch=head_branch,
            base_branch=base_branch,
        )
    except GitHubError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    number = pr.get("number")
    pr_url = (pr.get("html_url") or "").strip()
    status = (pr.get("state") or "").strip()
    details = {
        "repo": repo,
        "title": title,
        "head_branch": head_branch,
        "base_branch": base_branch,
        "number": number,
        "url": pr_url,
        "status": status,
    }
    action_id = _log_github_action(
        entity=entity,
        action_type="pr",
        repo=repo,
        details=details,
        session_id=payload.session_id,
    )
    card = {
        "type": "pr",
        "title": title,
        "branch": f"{head_branch}->{base_branch}",
        "url": pr_url,
    }
    _maybe_attach_card_to_thread(
        session_id=payload.session_id,
        entity=entity,
        title=f"Opened PR: {title}",
        card=card,
    )
    return JSONResponse(content={
        "success": True,
        "number": number,
        "url": pr_url,
        "status": status,
        "action_id": action_id,
        "chat_card": card,
    })


@github_router.get("/repos")
async def github_repos():
    client = _get_github_client()
    try:
        repos = client.list_repos()
    except GitHubError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    items = []
    for repo in repos:
        items.append({
            "name": repo.get("full_name") or repo.get("name"),
            "description": repo.get("description") or "",
            "default_branch": repo.get("default_branch") or "main",
            "last_updated": repo.get("updated_at"),
        })
    return JSONResponse(content={"repos": items})


@github_router.get("/repo/tree")
async def github_repo_tree(repo: str, branch: str = "main"):
    repo_name = _validate_repo_name(repo)
    branch_name = (branch or "main").strip()

    client = _get_github_client()
    _require_repo_access(client, repo_name, access="read")
    try:
        tree = client.get_repo_tree(repo=repo_name, branch=branch_name)
    except GitHubError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    entries = tree.get("tree") or []
    return JSONResponse(content={
        "repo": repo_name,
        "branch": branch_name,
        "entries": [
            {
                "path": e.get("path"),
                "type": e.get("type"),
                "size": e.get("size"),
                "sha": e.get("sha"),
                "url": e.get("url"),
            }
            for e in entries
        ],
    })


@github_router.get("/file")
async def github_file(repo: str, file_path: str, branch: str = "main"):
    repo_name = _validate_repo_name(repo)
    path = (file_path or "").strip().lstrip("/")
    branch_name = (branch or "main").strip()
    if not path:
        raise HTTPException(status_code=400, detail="file_path is required")

    client = _get_github_client()
    _require_repo_access(client, repo_name, access="read")
    try:
        file_obj = client.get_file(repo=repo_name, file_path=path, branch=branch_name)
    except GitHubError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    raw_content = file_obj.get("content") or ""
    encoding = (file_obj.get("encoding") or "").lower()
    decoded = ""
    if encoding == "base64" and raw_content:
        try:
            decoded = base64.b64decode(raw_content).decode("utf-8")
        except Exception:
            decoded = ""

    return JSONResponse(content={
        "repo": repo_name,
        "branch": branch_name,
        "file_path": path,
        "sha": file_obj.get("sha"),
        "content": decoded,
    })


@github_router.post("/issue")
async def github_issue(payload: GitHubIssueRequest):
    repo = _validate_repo_name(payload.repo)
    title = (payload.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    entity = _normalized_entity(payload.entity)
    labels = [str(lbl).strip() for lbl in payload.labels if str(lbl).strip()]

    client = _get_github_client()
    _require_repo_access(client, repo, access="issue_write")
    try:
        issue = client.create_issue(
            repo=repo,
            title=title,
            body=(payload.body or "").strip(),
            labels=labels,
        )
    except GitHubError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    number = issue.get("number")
    issue_url = (issue.get("html_url") or "").strip()
    details = {
        "repo": repo,
        "title": title,
        "number": number,
        "labels": labels,
        "url": issue_url,
        "status": issue.get("state"),
    }
    action_id = _log_github_action(
        entity=entity,
        action_type="issue",
        repo=repo,
        details=details,
        session_id=payload.session_id,
    )
    card = {
        "type": "issue",
        "title": title,
        "url": issue_url,
    }
    _maybe_attach_card_to_thread(
        session_id=payload.session_id,
        entity=entity,
        title=f"Opened Issue: {title}",
        card=card,
    )
    return JSONResponse(content={
        "success": True,
        "number": number,
        "url": issue_url,
        "status": issue.get("state"),
        "action_id": action_id,
        "chat_card": card,
    })

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start service immediately and load heavy models in background."""
    logger.info("Starting Sylana Vessel Server...")

    async def _bg_init():
        try:
            await asyncio.to_thread(load_models)
            await start_scheduler_if_needed()
        except Exception:
            logger.exception("Background model initialization failed")

    asyncio.create_task(_bg_init())
    yield
    # Cleanup
    if state.memory_manager:
        state.memory_manager.close()
    if state.relationship_db:
        state.relationship_db.close()
    global scheduler
    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)
    logger.info("Sylana Vessel Server shut down")


app = FastAPI(
    title="Sylana Vessel",
    description="AI Companion Soul Preservation System",
    version="1.0",
    lifespan=lifespan
)
app.include_router(github_router)
app.include_router(code_router)
app.include_router(sessions_router)
app.include_router(prospects_router)
app.include_router(email_drafts_router)
app.include_router(devices_router)
app.include_router(tracking_router)

# Cross-origin support for mobile/web clients hitting deployed API domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, "CORS_ORIGINS", ["*"]),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat interface"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Sylana Vessel</h1><p>static/index.html not found</p>")


@app.post("/api/chat")
@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint - returns streaming SSE response"""
    if not state.ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Models still loading. Please wait."}
        )

    body = await request.json()
    user_input = body.get("message", "").strip()
    thread_id = body.get("thread_id")
    requested_tools = body.get("active_tools", body.get("activeTools", None))
    personality = (body.get("personality") or "sylana").strip().lower()
    if state.personality_manager and personality not in state.personality_manager.list_personalities():
        personality = "sylana"

    if not user_input:
        return JSONResponse(
            status_code=400,
            content={"error": "No message provided"}
        )
    try:
        thread_id = int(thread_id) if thread_id is not None else None
    except Exception:
        thread_id = None
    if thread_id is not None and not _thread_exists(thread_id):
        thread_id = None
    if requested_tools is None:
        resolved_tools = _get_thread_tools(thread_id) if thread_id is not None else list(DEFAULT_ACTIVE_TOOLS)
    else:
        resolved_tools = normalize_active_tools(requested_tools)
    if thread_id is None:
        thread = create_chat_thread(title=f"[{personality}] {user_input[:80]}", active_tools=resolved_tools)
        thread_id = thread["id"]
    else:
        _set_thread_tools(thread_id, resolved_tools)

    logger.info(f"Chat request: {user_input[:50]}...")

    # Use streaming
    return EventSourceResponse(
        generate_response_stream(
            user_input,
            thread_id=thread_id,
            personality=personality,
            active_tools=resolved_tools,
        ),
        media_type="text/event-stream"
    )


@app.post("/api/chat/sync")
@app.post("/chat/sync")
async def chat_sync(request: Request):
    """Non-streaming chat endpoint"""
    if not state.ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Models still loading. Please wait."}
        )

    body = await request.json()
    user_input = body.get("message", "").strip()
    thread_id = body.get("thread_id")
    requested_tools = body.get("active_tools", body.get("activeTools", None))
    personality = (body.get("personality") or "sylana").strip().lower()
    if state.personality_manager and personality not in state.personality_manager.list_personalities():
        personality = "sylana"

    if not user_input:
        return JSONResponse(
            status_code=400,
            content={"error": "No message provided"}
        )
    try:
        thread_id = int(thread_id) if thread_id is not None else None
    except Exception:
        thread_id = None
    if thread_id is not None and not _thread_exists(thread_id):
        thread_id = None
    if requested_tools is None:
        resolved_tools = _get_thread_tools(thread_id) if thread_id is not None else list(DEFAULT_ACTIVE_TOOLS)
    else:
        resolved_tools = normalize_active_tools(requested_tools)
    if thread_id is None:
        thread = create_chat_thread(title=f"[{personality}] {user_input[:80]}", active_tools=resolved_tools)
        thread_id = thread["id"]
    else:
        _set_thread_tools(thread_id, resolved_tools)

    try:
        result = generate_response(
            user_input,
            thread_id=thread_id,
            personality=personality,
            active_tools=resolved_tools,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"chat_sync failed for thread_id={thread_id}: {e}")
        return _chat_sync_error_response(e, thread_id=thread_id)


@app.get("/api/threads")
async def threads(limit: int = 100):
    """List saved chat threads."""
    limit = max(1, min(limit, 300))
    return JSONResponse(content={"threads": list_chat_threads(limit=limit)})


@app.post("/api/threads")
async def create_thread(request: Request):
    """Create a new empty thread."""
    body = await request.json()
    title = (body.get("title") or "").strip()
    active_tools = normalize_active_tools(body.get("active_tools", body.get("activeTools", None)))
    thread = create_chat_thread(title=title or "New Thread", active_tools=active_tools)
    return JSONResponse(content=thread)


@app.get("/api/threads/{thread_id}/messages")
async def thread_messages(thread_id: int, limit: int = 300):
    """Load messages for one thread."""
    if not _thread_exists(thread_id):
        return JSONResponse(status_code=404, content={"error": "Thread not found"})
    limit = max(1, min(limit, 1000))
    thread = get_chat_thread(thread_id)
    return JSONResponse(content={
        "thread_id": thread_id,
        "active_tools": (thread or {}).get("active_tools", list(DEFAULT_ACTIVE_TOOLS)),
        "conversation_metadata": (thread or {}).get("conversation_metadata", {}),
        "messages": get_chat_messages(thread_id, limit=limit),
    })


@app.patch("/conversations/{conversation_id}/tools")
@app.patch("/api/conversations/{conversation_id}/tools")
async def update_conversation_tools(conversation_id: int, request: Request):
    if not _thread_exists(conversation_id):
        return JSONResponse(status_code=404, content={"error": "Conversation not found"})
    body = await request.json()
    active_tools = normalize_active_tools(body.get("active_tools", body.get("activeTools", None)))
    _set_thread_tools(conversation_id, active_tools)
    thread = get_chat_thread(conversation_id) or {}
    return JSONResponse(content={
        "conversation_id": conversation_id,
        "active_tools": thread.get("active_tools", active_tools),
        "conversation_metadata": thread.get("conversation_metadata", {}),
    })


@app.get("/tools/available")
@app.get("/api/tools/available")
async def tools_available():
    return JSONResponse(content={
        "default_active_tools": list(DEFAULT_ACTIVE_TOOLS),
        "tools": AVAILABLE_TOOLS,
    })


@app.get("/api/status")
async def status():
    """System status"""
    web_search_available = bool(
        state.claude_model
        and getattr(state.claude_model, "enable_web_search", False)
    )
    info = {
        "ready": state.ready,
        "model": config.CLAUDE_MODEL,
        "provider": "anthropic",
        "gpu": "N/A (API model)",
        "vram_gb": 0,
        "personality": "Multi",
        "personalities": state.personality_manager.list_personalities() if state.personality_manager else ["sylana", "claude"],
        "voice_validator": state.voice_validator is not None,
        "relationship_memory": state.relationship_db is not None,
        "emotion_model": f"API ({getattr(config, 'EMOTION_MODEL', 'gpt-4o-mini')})" if state.emotion_detector else "neutral-fallback",
        "web_search_enabled": web_search_available,
        "web_search_provider": "brave" if web_search_available else "disabled",
        "continuity_enabled": state.memory_manager is not None,
        "memory_encryption_enabled": bool(getattr(config, "MEMORY_ENCRYPTION_KEY", "")),
        "turns_this_session": state.turn_count
    }

    # Memory stats
    if state.memory_manager:
        info["memory"] = state.memory_manager.get_stats()

    # Relationship stats
    if state.relationship_db:
        info["relationship"] = state.relationship_db.get_stats()

    # Uptime
    if state.start_time:
        uptime = time.time() - state.start_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        info["uptime"] = f"{hours}h {minutes}m"

    return JSONResponse(content=info)


@app.get("/api/personalities")
@app.get("/personalities")
async def get_personalities():
    """Return available personalities."""
    personalities = ["sylana", "claude"]
    if state.personality_manager:
        personalities = state.personality_manager.list_personalities()
    return JSONResponse(content={
        "personalities": personalities,
        "default": "sylana",
    })


@app.get("/api/memories/search")
async def search_memories(q: str, k: int = 5, personality: str = "sylana"):
    """Search memories"""
    if not state.memory_manager:
        return JSONResponse(
            status_code=503,
            content={"error": "Memory system not ready"}
        )

    results = state.memory_manager.recall_relevant(q, k=k, personality=personality)
    return JSONResponse(content={
        "query": q,
        "personality": personality,
        "conversations": results.get('conversations', []),
        "core_memories": results.get('core_memories', [])
    })


@app.post("/api/memories/privacy")
async def set_memory_privacy(request: Request):
    """Update privacy level for a memory row."""
    body = await request.json()
    memory_id = body.get("memory_id")
    privacy_level = (body.get("privacy_level") or "").strip().lower()
    if privacy_level not in {"private", "shared", "public"}:
        return JSONResponse(status_code=400, content={"error": "privacy_level must be private|shared|public"})
    try:
        memory_id = int(memory_id)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "memory_id must be an integer"})

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE memories SET privacy_level = %s WHERE id = %s RETURNING id",
            (privacy_level, memory_id),
        )
        row = cur.fetchone()
        conn.commit()
        if not row:
            return JSONResponse(status_code=404, content={"error": "Memory not found"})
        return JSONResponse(content={"ok": True, "memory_id": memory_id, "privacy_level": privacy_level})
    except Exception as e:
        _safe_rollback(conn, "set_memory_privacy")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: int):
    """Delete a memory row for privacy lifecycle control."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM memories WHERE id = %s RETURNING id", (memory_id,))
        row = cur.fetchone()
        conn.commit()
        if not row:
            return JSONResponse(status_code=404, content={"error": "Memory not found"})
        return JSONResponse(content={"ok": True, "deleted_memory_id": memory_id})
    except Exception as e:
        _safe_rollback(conn, "delete_memory")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/memories/backup")
async def backup_memories(request: Request):
    """Write a memory integrity snapshot to local disk."""
    if not state.memory_manager:
        return JSONResponse(status_code=503, content={"error": "Memory system not ready"})
    body = await request.json()
    out_path = (body.get("path") or "data/memory_snapshot.json").strip()
    result = state.memory_manager.backup_memory_integrity(out_path)
    code = 200 if result.get("ok") else 500
    return JSONResponse(status_code=code, content=result)


@app.post("/api/memories/recover")
async def recover_memories(request: Request):
    """Recover memory rows from a snapshot file."""
    if not state.memory_manager:
        return JSONResponse(status_code=503, content={"error": "Memory system not ready"})
    body = await request.json()
    in_path = (body.get("path") or "").strip()
    personality = (body.get("personality") or "sylana").strip().lower()
    if not in_path:
        return JSONResponse(status_code=400, content={"error": "path is required"})
    result = state.memory_manager.recover_memory_integrity(in_path, personality=personality)
    code = 200 if result.get("ok") else 500
    return JSONResponse(status_code=code, content=result)


@app.get("/api/health")
async def health():
    """Simple health check"""
    return {"status": "alive", "ready": state.ready}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT") or os.environ.get("SERVER_PORT") or 10000)
    host = os.environ.get("SERVER_HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
