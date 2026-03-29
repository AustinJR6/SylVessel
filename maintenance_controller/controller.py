from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import psycopg2
import psycopg2.extras


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(__file__).with_name("repair_output.schema.json")
DEFAULT_REPO_PROFILES = {
    "AustinJR6/SylVessel": {
        "base_branch": "main",
        "verify_commands": [
            ["python", "-m", "py_compile", "server.py"],
            ["python", "-m", "unittest", "discover", "-s", "tests", "-v"],
        ],
    },
    "AustinJR6/SoulsApp": {
        "base_branch": "main",
        "verify_commands": [
            ["npx", "tsc", "--noEmit"],
        ],
    },
}
FAILED_ACTION_CONCLUSIONS = {"failure", "timed_out", "startup_failure"}
REPAIR_MODES = {"diagnosis", "proposal"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def env(name: str, default: str = "") -> str:
    return str(os.getenv(name) or default).strip()


def normalize_repair_mode(value: str, default: str = "diagnosis") -> str:
    mode = str(value or default).strip().lower()
    return mode if mode in REPAIR_MODES else default


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def run_command(command: List[str], *, cwd: Optional[Path] = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class GitHubError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        self.status_code = int(status_code)
        self.message = message
        super().__init__(message)


class GitHubClient:
    def __init__(self, token: str):
        self.token = token.strip()
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN is required")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sylana-maintenance-controller",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        expected: Optional[set[int]] = None,
    ) -> Dict[str, Any]:
        expected_codes = expected or {200}
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        data = None
        headers = dict(self.headers)
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url=url, headers=headers, data=data, method=method.upper())
        try:
            with urlopen(req, timeout=30) as response:
                status = int(response.status)
                body = response.read().decode("utf-8") if response else ""
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            message = f"GitHub API error ({exc.code})"
            if body:
                try:
                    message = (json.loads(body) or {}).get("message") or message
                except Exception:
                    message = body[:200]
            raise GitHubError(exc.code, message) from exc
        except URLError as exc:
            raise GitHubError(503, f"GitHub unreachable: {exc.reason}") from exc
        if status not in expected_codes:
            raise GitHubError(status, f"Unexpected status {status}")
        if not body:
            return {}
        return json.loads(body)

    def list_workflow_runs(self, repo: str, *, branch: Optional[str] = None, status: Optional[str] = None, per_page: int = 20) -> Dict[str, Any]:
        params = [f"per_page={max(1, min(int(per_page), 100))}"]
        if branch:
            params.append(f"branch={quote(branch, safe='')}")
        if status:
            params.append(f"status={quote(status, safe='')}")
        return self._request("GET", f"/repos/{repo}/actions/runs", query="&".join(params), expected={200})

    def get_pull_request(self, repo: str, number: int) -> Dict[str, Any]:
        return self._request("GET", f"/repos/{repo}/pulls/{int(number)}", expected={200})

    def create_pull_request(self, repo: str, *, title: str, body: str, head_branch: str, base_branch: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/repos/{repo}/pulls",
            payload={"title": title, "body": body, "head": head_branch, "base": base_branch},
            expected={201},
        )


@dataclass
class ControllerConfig:
    backend_repo: str
    app_repo: str
    vessel_api_base_url: str
    maintenance_read_token: str
    github_token: str
    clone_root: Path
    poll_seconds: int
    codex_bin: str
    codex_model: str
    repair_mode: str

    @classmethod
    def from_env(cls) -> "ControllerConfig":
        clone_root = Path(env("MAINTENANCE_CLONE_ROOT", str(ROOT / "data" / "maintenance_workspaces")))
        clone_root.mkdir(parents=True, exist_ok=True)
        return cls(
            backend_repo=env("MAINTENANCE_BACKEND_REPO", "AustinJR6/SylVessel"),
            app_repo=env("MAINTENANCE_APP_REPO", "AustinJR6/SoulsApp"),
            vessel_api_base_url=env("VESSEL_API_BASE_URL", "https://167-99-127-14.sslip.io").rstrip("/"),
            maintenance_read_token=env("MAINTENANCE_READ_TOKEN", env("MAINTENANCE_CONTROLLER_TOKEN")),
            github_token=env("GITHUB_TOKEN"),
            clone_root=clone_root,
            poll_seconds=max(30, int(env("MAINTENANCE_POLL_SECONDS", "300"))),
            codex_bin=env("MAINTENANCE_CODEX_BIN", "codex"),
            codex_model=env("MAINTENANCE_CODEX_MODEL", "gpt-5.4"),
            repair_mode=normalize_repair_mode(env("MAINTENANCE_REPAIR_MODE", "diagnosis")),
        )


class RepairStore:
    def __init__(self, db_url: str):
        if not db_url:
            raise RuntimeError("SUPABASE_DB_URL is required")
        self.db_url = db_url

    def connect(self):
        conn = psycopg2.connect(self.db_url, connect_timeout=10)
        conn.autocommit = False
        return conn

    def _serialize_incident(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return json_safe(dict(row or {}))

    def _insert_event(
        self,
        cur: psycopg2.extensions.cursor,
        *,
        incident_id: Optional[str],
        run_id: Optional[str] = None,
        event_kind: str,
        actor: str = "maintenance-controller",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        cur.execute(
            """
            INSERT INTO repair_events (incident_id, run_id, event_kind, actor, metadata)
            VALUES (%s::uuid, %s::uuid, %s, %s, %s::jsonb)
            """,
            (
                incident_id or None,
                run_id or None,
                str(event_kind or "").strip() or "event",
                str(actor or "maintenance-controller").strip() or "maintenance-controller",
                json.dumps(metadata or {}, ensure_ascii=True),
            ),
        )

    def upsert_incident(
        self,
        *,
        dedupe_key: str,
        repo: str,
        environment: str,
        source: str,
        severity: str,
        symptom_summary: str,
        reproduction_hints: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        conn = self.connect()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute(
                """
                INSERT INTO repair_incidents (
                    dedupe_key, repo, environment, source, severity, status,
                    symptom_summary, reproduction_hints, metadata
                )
                VALUES (%s, %s, %s, %s, %s, 'detected', %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (dedupe_key) DO UPDATE
                SET repo = EXCLUDED.repo,
                    environment = EXCLUDED.environment,
                    source = EXCLUDED.source,
                    severity = EXCLUDED.severity,
                    symptom_summary = EXCLUDED.symptom_summary,
                    reproduction_hints = EXCLUDED.reproduction_hints,
                    metadata = repair_incidents.metadata || EXCLUDED.metadata,
                    last_seen_at = NOW()
                RETURNING incident_id, dedupe_key, repo, environment, source, severity, status,
                          symptom_summary, reproduction_hints, root_cause_summary, metadata, pr_url,
                          latest_run_id, first_seen_at, last_seen_at, resolved_at
                """,
                (
                    dedupe_key,
                    repo,
                    environment,
                    source,
                    severity,
                    symptom_summary,
                    json.dumps(reproduction_hints or {}, ensure_ascii=True),
                    json.dumps(metadata or {}, ensure_ascii=True),
                ),
            )
            row = self._serialize_incident(dict(cur.fetchone() or {}))
            self._insert_event(
                cur,
                incident_id=str(row.get("incident_id") or ""),
                event_kind="incident_detected",
                metadata={
                    "dedupe_key": dedupe_key,
                    "repo": repo,
                    "environment": environment,
                    "source": source,
                    "severity": severity,
                    "symptom_summary": symptom_summary,
                },
            )
            conn.commit()
            return row
        finally:
            conn.close()

    def create_run(self, *, incident_id: str, repo: str, environment: str, workspace_path: str, base_branch: str) -> str:
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO repair_runs (incident_id, repo, environment, workspace_path, base_branch, status)
                VALUES (%s::uuid, %s, %s, %s, %s, 'investigating')
                RETURNING run_id
                """,
                (incident_id, repo, environment, workspace_path, base_branch),
            )
            run_id = str(cur.fetchone()[0])
            cur.execute(
                "UPDATE repair_incidents SET status = 'investigating', latest_run_id = %s::uuid, last_seen_at = NOW() WHERE incident_id = %s::uuid",
                (run_id, incident_id),
            )
            self._insert_event(
                cur,
                incident_id=incident_id,
                run_id=run_id,
                event_kind="run_started",
                metadata={
                    "repo": repo,
                    "environment": environment,
                    "workspace_path": workspace_path,
                    "base_branch": base_branch,
                },
            )
            conn.commit()
            return run_id
        finally:
            conn.close()

    def update_run_proposal(
        self,
        *,
        run_id: str,
        branch_name: str,
        pr_number: int,
        pr_url: str,
        root_cause_summary: str,
        patch_summary: str,
        verification_summary: str,
        risk_level: str,
        artifact_payload: Dict[str, Any],
        note_id: Optional[str],
    ) -> None:
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE repair_runs
                SET branch_name = %s,
                    status = 'proposed',
                    pr_number = %s,
                    pr_url = %s,
                    root_cause_summary = %s,
                    patch_summary = %s,
                    verification_summary = %s,
                    risk_level = %s,
                    artifact_payload = %s::jsonb,
                    note_id = %s::uuid,
                    updated_at = NOW()
                WHERE run_id = %s::uuid
                RETURNING incident_id
                """,
                (
                    branch_name,
                    pr_number,
                    pr_url,
                    root_cause_summary,
                    patch_summary,
                    verification_summary,
                    risk_level,
                    json.dumps(artifact_payload or {}, ensure_ascii=True),
                    note_id,
                    run_id,
                ),
            )
            row = cur.fetchone()
            incident_id = str(row[0]) if row else ""
            if incident_id:
                cur.execute(
                    """
                    UPDATE repair_incidents
                    SET status = 'proposed',
                        root_cause_summary = %s,
                        pr_url = %s,
                        latest_run_id = %s::uuid,
                        last_seen_at = NOW()
                    WHERE incident_id = %s::uuid
                    """,
                    (root_cause_summary, pr_url, run_id, incident_id),
                )
                self._insert_event(
                    cur,
                    incident_id=incident_id,
                    run_id=run_id,
                    event_kind="proposal_created",
                    metadata={
                        "branch_name": branch_name,
                        "pr_number": pr_number,
                        "pr_url": pr_url,
                        "risk_level": risk_level,
                        "verification_summary": verification_summary,
                        "artifact_payload": artifact_payload or {},
                    },
                )
            conn.commit()
        finally:
            conn.close()

    def fail_run(self, *, run_id: str, reason: str) -> None:
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE repair_runs
                SET status = 'failed', failure_reason = %s, updated_at = NOW()
                WHERE run_id = %s::uuid
                RETURNING incident_id
                """,
                (reason, run_id),
            )
            row = cur.fetchone()
            if row:
                incident_id = str(row[0])
                cur.execute(
                    """
                    UPDATE repair_incidents
                    SET status = 'failed', last_seen_at = NOW()
                    WHERE incident_id = %s::uuid
                    """,
                    (incident_id,),
                )
                self._insert_event(
                    cur,
                    incident_id=incident_id,
                    run_id=run_id,
                    event_kind="run_failed",
                    metadata={"reason": reason},
                )
            conn.commit()
        finally:
            conn.close()

    def diagnose_run(
        self,
        *,
        run_id: str,
        root_cause_summary: str,
        verification_summary: str = "",
        artifact_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE repair_runs
                SET status = 'diagnosed',
                    root_cause_summary = %s,
                    verification_summary = %s,
                    artifact_payload = %s::jsonb,
                    updated_at = NOW()
                WHERE run_id = %s::uuid
                RETURNING incident_id
                """,
                (
                    root_cause_summary,
                    verification_summary,
                    json.dumps(artifact_payload or {}, ensure_ascii=True),
                    run_id,
                ),
            )
            row = cur.fetchone()
            if row:
                incident_id = str(row[0])
                cur.execute(
                    """
                    UPDATE repair_incidents
                    SET status = 'investigating',
                        root_cause_summary = %s,
                        last_seen_at = NOW(),
                        latest_run_id = %s::uuid
                    WHERE incident_id = %s::uuid
                    """,
                    (root_cause_summary, run_id, incident_id),
                )
                self._insert_event(
                    cur,
                    incident_id=incident_id,
                    run_id=run_id,
                    event_kind="run_diagnosed",
                    metadata={
                        "root_cause_summary": root_cause_summary,
                        "verification_summary": verification_summary,
                        "artifact_payload": artifact_payload or {},
                    },
                )
            conn.commit()
        finally:
            conn.close()

    def note_for_proposal(
        self,
        *,
        run_id: str,
        incident: Dict[str, Any],
        repo: str,
        pr_url: str,
        risk_level: str,
        verification_summary: str,
        root_cause_summary: str,
    ) -> str:
        note_id = str(uuid.uuid4())
        metadata = {
            "note_kind": "follow_up",
            "why_now": "A repair proposal is ready for your review before merge and deploy.",
            "topic_key": f"repair:{incident.get('incident_id')}",
            "importance_score": 0.82,
            "durable": True,
            "surface_kind": "approval",
            "action_kind": "repair_pr",
            "action_payload": {"repair_run_id": run_id, "repo": repo, "pr_url": pr_url},
            "route_target": pr_url,
            "delivery_policy": "rare_push",
            "confidence_score": 0.78,
            "personality": "claude",
            "repo": repo,
            "incident_id": incident.get("incident_id"),
            "repair_run_id": run_id,
            "pr_url": pr_url,
            "verification_summary": verification_summary,
            "risk_level": risk_level,
            "symptom_summary": incident.get("symptom_summary") or "",
            "root_cause_summary": root_cause_summary,
        }
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO proactive_notes (
                    note_id, source, source_id, title, body, severity, status, dedupe_key,
                    announce_policy, requires_approval, approval_status, metadata
                )
                VALUES (
                    %s::uuid, 'maintenance_controller', %s, %s, %s, 'warning', 'pending', %s,
                    'important_only', TRUE, 'pending_approval', %s::jsonb
                )
                """,
                (
                    note_id,
                    run_id,
                    f"Repair proposal for {repo}",
                    verification_summary or (incident.get("symptom_summary") or "Repair proposal ready for review."),
                    f"repair-proposal:{run_id}",
                    json.dumps(metadata, ensure_ascii=True),
                ),
            )
            cur.execute(
                """
                INSERT INTO proactive_note_events (note_id, event_kind, actor, metadata)
                VALUES (%s::uuid, 'created', 'maintenance-controller', %s::jsonb)
                """,
                (
                    note_id,
                    json.dumps({"repair_run_id": run_id, "repo": repo, "pr_url": pr_url}, ensure_ascii=True),
                ),
            )
            self._insert_event(
                cur,
                incident_id=str(incident.get("incident_id") or ""),
                run_id=run_id,
                event_kind="review_note_created",
                metadata={"note_id": note_id, "repo": repo, "pr_url": pr_url},
            )
            conn.commit()
            return note_id
        finally:
            conn.close()

    def list_active_runs(self) -> List[Dict[str, Any]]:
        conn = self.connect()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute(
                """
                SELECT run_id, incident_id, repo, status, pr_number, pr_url
                FROM repair_runs
                WHERE status IN ('proposed', 'approved', 'merged')
                ORDER BY updated_at DESC
                LIMIT 50
                """
            )
            rows = [dict(row) for row in cur.fetchall()]
            conn.commit()
            return rows
        finally:
            conn.close()

    def list_requested_investigations(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        conn = self.connect()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute(
                """
                WITH latest_requests AS (
                    SELECT DISTINCT ON (e.incident_id)
                        e.incident_id,
                        e.created_at AS requested_at
                    FROM repair_events e
                    JOIN repair_incidents i ON i.incident_id = e.incident_id
                    WHERE e.event_kind = 'investigate_requested'
                      AND COALESCE(i.status, 'detected') NOT IN ('proposed', 'approved', 'merged', 'deployed', 'rejected')
                    ORDER BY e.incident_id, e.created_at DESC
                )
                SELECT
                    i.incident_id,
                    i.dedupe_key,
                    i.repo,
                    i.environment,
                    i.source,
                    i.severity,
                    i.status,
                    i.symptom_summary,
                    i.reproduction_hints,
                    i.root_cause_summary,
                    i.metadata,
                    i.pr_url,
                    i.latest_run_id,
                    i.first_seen_at,
                    i.last_seen_at,
                    i.resolved_at,
                    lr.requested_at
                FROM latest_requests lr
                JOIN repair_incidents i ON i.incident_id = lr.incident_id
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM repair_runs r
                    WHERE r.incident_id = i.incident_id
                      AND r.created_at >= lr.requested_at
                )
                ORDER BY lr.requested_at ASC
                LIMIT %s
                """,
                (max(1, min(int(limit), 50)),),
            )
            rows = [self._serialize_incident(dict(row)) for row in cur.fetchall()]
            conn.commit()
            return rows
        finally:
            conn.close()

    def set_run_status(self, *, run_id: str, status: str, extra_field_sql: str = "", params: Optional[List[Any]] = None) -> None:
        conn = self.connect()
        cur = conn.cursor()
        params = params or []
        try:
            cur.execute(
                f"UPDATE repair_runs SET status = %s, updated_at = NOW(){extra_field_sql} WHERE run_id = %s::uuid RETURNING incident_id",
                tuple([status, *params, run_id]),
            )
            row = cur.fetchone()
            if row:
                incident_status = "merged" if status == "merged" else "deployed" if status == "deployed" else "approved"
                incident_id = str(row[0])
                cur.execute(
                    "UPDATE repair_incidents SET status = %s, last_seen_at = NOW(), latest_run_id = %s::uuid WHERE incident_id = %s::uuid",
                    (incident_status, run_id, incident_id),
                )
                self._insert_event(
                    cur,
                    incident_id=incident_id,
                    run_id=run_id,
                    event_kind=f"run_{status}",
                    metadata={"status": status},
                )
            conn.commit()
        finally:
            conn.close()


class MaintenanceController:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.github = GitHubClient(cfg.github_token)
        self.store = RepairStore(env("SUPABASE_DB_URL"))

    def _should_auto_propose(self, incident: Dict[str, Any]) -> bool:
        if self.cfg.repair_mode != "proposal":
            return False
        source = str(incident.get("source") or "").strip().lower()
        return source == "github_actions"

    def _request_json(self, method: str, path: str, *, payload: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        url = f"{self.cfg.vessel_api_base_url}{path}"
        merged_headers = {"Accept": "application/json", "User-Agent": "sylana-maintenance-controller"}
        if headers:
            merged_headers.update(headers)
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            merged_headers["Content-Type"] = "application/json"
        req = Request(url=url, headers=merged_headers, data=data, method=method.upper())
        with urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}

    def _workflow_run_signals_deployed(self, repo: str, pr: Dict[str, Any]) -> bool:
        merge_sha = str(pr.get("merge_commit_sha") or "").strip()
        merged_at = str(pr.get("merged_at") or "").strip()
        if not merge_sha and not merged_at:
            return False
        try:
            payload = self.github.list_workflow_runs(repo, status="completed", per_page=20)
        except Exception:
            return False
        for run in payload.get("workflow_runs") or []:
            if str(run.get("conclusion") or "").strip().lower() != "success":
                continue
            run_name = str(run.get("name") or "").strip().lower()
            head_sha = str(run.get("head_sha") or "").strip()
            created_at = str(run.get("created_at") or "").strip()
            if merge_sha and head_sha == merge_sha:
                return True
            if merged_at and created_at and created_at >= merged_at:
                if repo == self.cfg.backend_repo and "deploy" in run_name:
                    return True
                if repo == self.cfg.app_repo and ("preview" in run_name or "eas" in run_name or "android" in run_name):
                    return True
        return False

    def detect_health_incident(self) -> None:
        try:
            payload = self._request_json("GET", "/api/health")
            if payload.get("ready"):
                return
            symptom = "Vessel health endpoint is reachable but not ready"
            metadata = {"payload": payload}
        except Exception as exc:
            symptom = f"Vessel health endpoint is unavailable: {exc}"
            metadata = {"error": str(exc)}
        self.store.upsert_incident(
            dedupe_key=f"health:{self.cfg.vessel_api_base_url}",
            repo=self.cfg.backend_repo,
            environment="production",
            source="healthcheck",
            severity="critical",
            symptom_summary=symptom,
            reproduction_hints={"endpoint": f"{self.cfg.vessel_api_base_url}/api/health"},
            metadata=metadata,
        )

    def detect_failed_workflow_runs(self, repo: str) -> List[Dict[str, Any]]:
        payload = self.github.list_workflow_runs(repo, status="completed", per_page=15)
        incidents: List[Dict[str, Any]] = []
        for run in payload.get("workflow_runs") or []:
            conclusion = str(run.get("conclusion") or "").strip().lower()
            if conclusion not in FAILED_ACTION_CONCLUSIONS:
                continue
            dedupe_key = f"gh-run:{repo}:{run.get('id')}"
            incident = self.store.upsert_incident(
                dedupe_key=dedupe_key,
                repo=repo,
                environment="production",
                source="github_actions",
                severity="critical" if repo == self.cfg.backend_repo else "warning",
                symptom_summary=f"GitHub Actions failed: {run.get('name') or 'workflow'}",
                reproduction_hints={
                    "html_url": run.get("html_url"),
                    "run_id": run.get("id"),
                    "workflow_name": run.get("name"),
                    "head_branch": run.get("head_branch"),
                    "head_sha": run.get("head_sha"),
                    "conclusion": conclusion,
                },
                metadata={"workflow_run": run},
            )
            incidents.append(incident)
        return incidents

    def fetch_log_tail(self) -> Dict[str, Any]:
        if not self.cfg.maintenance_read_token:
            return {"lines": []}
        try:
            return self._request_json(
                "GET",
                "/repairs/logs/tail?lines=120",
                headers={"X-Maintenance-Token": self.cfg.maintenance_read_token},
            )
        except Exception:
            return {"lines": []}

    def process_requested_investigations(self) -> None:
        for incident in self.store.list_requested_investigations(limit=10):
            self.propose_fix(incident)

    def clone_repo(self, repo: str, workspace: Path) -> None:
        clone_url = f"https://x-access-token:{self.cfg.github_token}@github.com/{repo}.git"
        result = run_command(["git", "clone", clone_url, str(workspace)], timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip() or result.stdout.strip()}")

    def run_codex(self, workspace: Path, incident: Dict[str, Any]) -> Dict[str, Any]:
        output_path = workspace / "repair-output.json"
        prompt = f"""
Investigate and, only if safe and clearly attributable, patch this incident in the current repository clone.

Incident:
{json.dumps(json_safe(incident), indent=2)}

Hard rules:
- Never edit secrets or infrastructure outside tracked repo files.
- Never perform destructive database changes.
- If the issue is ambiguous, non-deterministic, or infra-related, do diagnosis only and do not patch.
- If you do patch, run the repo-appropriate verification commands.
- Leave the repo on the current branch with any proposed changes staged in the working tree.
- Return JSON matching the provided schema.
""".strip()
        command = [
            self.cfg.codex_bin,
            "exec",
            "--model",
            self.cfg.codex_model,
            "--dangerously-bypass-approvals-and-sandbox",
            "--output-schema",
            str(SCHEMA_PATH),
            "--output-last-message",
            str(output_path),
            prompt,
        ]
        result = run_command(command, cwd=workspace, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"codex exec failed: {result.stderr.strip() or result.stdout.strip()}")
        if not output_path.exists():
            raise RuntimeError("codex exec did not produce repair output")
        return json.loads(output_path.read_text(encoding="utf-8"))

    def verify_workspace(self, repo: str, workspace: Path) -> List[str]:
        commands = DEFAULT_REPO_PROFILES.get(repo, {}).get("verify_commands") or []
        executed: List[str] = []
        for command in commands:
            result = run_command(command, cwd=workspace, timeout=3600)
            executed.append(" ".join(command))
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"Verification failed: {' '.join(command)}")
        return executed

    def create_branch_and_pr(self, repo: str, incident: Dict[str, Any], run_id: str, workspace: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in (incident.get("symptom_summary") or "repair"))[:42].strip("-") or "repair"
        branch = f"repair/{slug}-{run_id[:8]}"
        base_branch = DEFAULT_REPO_PROFILES.get(repo, {}).get("base_branch", "main")
        checkout = run_command(["git", "checkout", "-b", branch, f"origin/{base_branch}"], cwd=workspace, timeout=300)
        if checkout.returncode != 0:
            raise RuntimeError(checkout.stderr.strip() or checkout.stdout.strip() or "git checkout failed")
        add_result = run_command(["git", "add", "-A"], cwd=workspace, timeout=300)
        if add_result.returncode != 0:
            raise RuntimeError(add_result.stderr.strip() or add_result.stdout.strip() or "git add failed")
        diff_result = run_command(["git", "status", "--short"], cwd=workspace, timeout=300)
        if diff_result.returncode != 0 or not diff_result.stdout.strip():
            raise RuntimeError("No tracked changes were produced for this repair run")
        commit_message = f"repair: {incident.get('symptom_summary') or 'maintenance fix'}"
        commit = run_command(["git", "commit", "-m", commit_message], cwd=workspace, timeout=600)
        if commit.returncode != 0:
            raise RuntimeError(commit.stderr.strip() or commit.stdout.strip() or "git commit failed")
        push = run_command(["git", "push", "origin", branch], cwd=workspace, timeout=1800)
        if push.returncode != 0:
            raise RuntimeError(push.stderr.strip() or push.stdout.strip() or "git push failed")
        pr = self.github.create_pull_request(
            repo,
            title=f"repair: {incident.get('symptom_summary') or 'maintenance fix'}",
            body=(
                f"## Root Cause\n{summary.get('root_cause_summary') or 'n/a'}\n\n"
                f"## Patch Summary\n{summary.get('patch_summary') or 'n/a'}\n\n"
                f"## Verification\n{summary.get('verification_summary') or 'n/a'}"
            ),
            head_branch=branch,
            base_branch=base_branch,
        )
        return {"branch_name": branch, "base_branch": base_branch, "pr": pr}

    def propose_fix(self, incident: Dict[str, Any]) -> None:
        repo = str(incident.get("repo") or "").strip()
        if not repo:
            return
        workspace = Path(tempfile.mkdtemp(prefix="repair-", dir=str(self.cfg.clone_root)))
        run_id = self.store.create_run(
            incident_id=str(incident.get("incident_id")),
            repo=repo,
            environment=str(incident.get("environment") or "production"),
            workspace_path=str(workspace),
            base_branch=DEFAULT_REPO_PROFILES.get(repo, {}).get("base_branch", "main"),
        )
        try:
            self.clone_repo(repo, workspace)
            summary = self.run_codex(workspace, incident)
            if bool(summary.get("diagnosis_only")):
                self.store.diagnose_run(
                    run_id=run_id,
                    root_cause_summary=str(summary.get("root_cause_summary") or "Repair requires human diagnosis only"),
                    verification_summary=str(summary.get("verification_summary") or ""),
                    artifact_payload={
                        "changed_files": summary.get("changed_files") or [],
                        "tests_run": summary.get("tests_run") or [],
                        "notes": summary.get("notes") or "",
                    },
                )
                return
            tests_run = self.verify_workspace(repo, workspace)
            summary["tests_run"] = tests_run
            proposal = self.create_branch_and_pr(repo, incident, run_id, workspace, summary)
            pr = proposal["pr"]
            note_id = self.store.note_for_proposal(
                run_id=run_id,
                incident=incident,
                repo=repo,
                pr_url=str(pr.get("html_url") or ""),
                risk_level=str(summary.get("risk_level") or "medium"),
                verification_summary=str(summary.get("verification_summary") or ""),
                root_cause_summary=str(summary.get("root_cause_summary") or ""),
            )
            self.store.update_run_proposal(
                run_id=run_id,
                branch_name=str(proposal["branch_name"]),
                pr_number=int(pr.get("number")),
                pr_url=str(pr.get("html_url") or ""),
                root_cause_summary=str(summary.get("root_cause_summary") or ""),
                patch_summary=str(summary.get("patch_summary") or ""),
                verification_summary=str(summary.get("verification_summary") or ""),
                risk_level=str(summary.get("risk_level") or "medium"),
                artifact_payload={
                    "changed_files": summary.get("changed_files") or [],
                    "tests_run": summary.get("tests_run") or [],
                    "notes": summary.get("notes") or "",
                },
                note_id=note_id,
            )
        except Exception as exc:
            self.store.fail_run(run_id=run_id, reason=str(exc))
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def sync_active_runs(self) -> None:
        for run in self.store.list_active_runs():
            repo = str(run.get("repo") or "").strip()
            pr_number = run.get("pr_number")
            if not repo or pr_number is None:
                continue
            try:
                pr = self.github.get_pull_request(repo, int(pr_number))
            except Exception:
                continue
            if pr.get("merged_at") and str(run.get("status")) != "merged":
                self.store.set_run_status(run_id=str(run.get("run_id")), status="merged", extra_field_sql=", merged_at = NOW()")
                run["status"] = "merged"
            if pr.get("merged_at") and str(run.get("status")) in {"merged", "approved"} and self._workflow_run_signals_deployed(repo, pr):
                self.store.set_run_status(run_id=str(run.get("run_id")), status="deployed", extra_field_sql=", deployed_at = NOW()")

    def run_once(self) -> None:
        self.detect_health_incident()
        incidents = []
        incidents.extend(self.detect_failed_workflow_runs(self.cfg.backend_repo))
        incidents.extend(self.detect_failed_workflow_runs(self.cfg.app_repo))
        self.fetch_log_tail()
        for incident in incidents:
            if self._should_auto_propose(incident):
                self.propose_fix(incident)
        self.process_requested_investigations()
        self.sync_active_runs()

    def loop(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.cfg.poll_seconds)


def main(argv: List[str]) -> int:
    cfg = ControllerConfig.from_env()
    controller = MaintenanceController(cfg)
    command = argv[1] if len(argv) > 1 else "run-once"
    if command == "loop":
        controller.loop()
        return 0
    if command == "run-once":
        controller.run_once()
        return 0
    print("Usage: python -m maintenance_controller.controller [run-once|loop]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
