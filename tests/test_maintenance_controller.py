from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from maintenance_controller import controller as maintenance_controller


class MaintenanceControllerTests(unittest.TestCase):
    def _build_controller(self):
        cfg = maintenance_controller.ControllerConfig(
            backend_repo="AustinJR6/SylVessel",
            app_repo="AustinJR6/SoulsApp",
            vessel_api_base_url="https://example.test",
            maintenance_read_token="token",
            github_token="github-token",
            clone_root=Path(tempfile.mkdtemp(prefix="maint-controller-")),
            poll_seconds=300,
            codex_bin="codex",
            codex_model="gpt-5.4",
            repair_mode="diagnosis",
        )
        controller = maintenance_controller.MaintenanceController.__new__(maintenance_controller.MaintenanceController)
        controller.cfg = cfg
        controller.github = SimpleNamespace()
        controller.store = SimpleNamespace()
        return controller

    def test_sync_active_runs_marks_merged_and_deployed(self):
        controller = self._build_controller()
        active_runs = [{"run_id": "run-1", "repo": "AustinJR6/SylVessel", "status": "approved", "pr_number": 17}]
        status_calls = []
        controller.store.list_active_runs = lambda: active_runs
        controller.store.set_run_status = lambda **kwargs: status_calls.append(kwargs)
        controller.github.get_pull_request = lambda repo, number: {
            "merged_at": "2026-03-29T12:00:00Z",
            "merge_commit_sha": "abc123",
        }
        controller.github.list_workflow_runs = lambda repo, status=None, per_page=20: {
            "workflow_runs": [
                {
                    "name": "Deploy to DigitalOcean",
                    "conclusion": "success",
                    "head_sha": "abc123",
                    "created_at": "2026-03-29T12:03:00Z",
                }
            ]
        }

        controller.sync_active_runs()

        self.assertEqual([call["status"] for call in status_calls], ["merged", "deployed"])

    def test_propose_fix_records_diagnosis_without_pr_when_codex_returns_diagnosis_only(self):
        controller = self._build_controller()
        controller.store.create_run = lambda **kwargs: "run-1"
        diagnosis_calls = []
        controller.store.diagnose_run = lambda **kwargs: diagnosis_calls.append(kwargs)
        controller.store.fail_run = lambda **kwargs: (_ for _ in ()).throw(AssertionError("fail_run should not run for diagnosis-only output"))
        controller.clone_repo = lambda repo, workspace: None
        controller.run_codex = lambda workspace, incident: {
            "diagnosis_only": True,
            "root_cause_summary": "The incident is ambiguous and should stay diagnosis-only.",
        }
        controller.verify_workspace = lambda repo, workspace: (_ for _ in ()).throw(AssertionError("verify_workspace should not run"))
        controller.create_branch_and_pr = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("create_branch_and_pr should not run"))
        controller.store.note_for_proposal = lambda **kwargs: (_ for _ in ()).throw(AssertionError("note_for_proposal should not run"))
        controller.store.update_run_proposal = lambda **kwargs: (_ for _ in ()).throw(AssertionError("update_run_proposal should not run"))

        controller.propose_fix(
            {
                "incident_id": "inc-1",
                "repo": "AustinJR6/SylVessel",
                "environment": "production",
                "symptom_summary": "Health endpoint unreachable",
            }
        )

        self.assertEqual(len(diagnosis_calls), 1)
        self.assertEqual(diagnosis_calls[0]["run_id"], "run-1")
        self.assertIn("diagnosis", diagnosis_calls[0]["root_cause_summary"].lower())

    def test_run_once_diagnosis_mode_does_not_auto_propose_failed_workflows(self):
        controller = self._build_controller()
        controller.cfg.repair_mode = "diagnosis"
        controller.detect_health_incident = lambda: None
        controller.detect_failed_workflow_runs = lambda repo: [
            {"incident_id": f"inc-{repo}", "repo": repo, "source": "github_actions", "symptom_summary": "Deploy failed"}
        ]
        controller.fetch_log_tail = lambda: {"lines": []}
        proposed = []
        controller.propose_fix = lambda incident: proposed.append(incident)
        controller.process_requested_investigations = lambda: None
        controller.sync_resolved_workflow_incidents = lambda: None
        controller.sync_active_runs = lambda: None

        controller.run_once()

        self.assertEqual(proposed, [])

    def test_process_requested_investigations_proposes_requested_incidents(self):
        controller = self._build_controller()
        incidents = [
            {"incident_id": "inc-1", "repo": "AustinJR6/SylVessel", "source": "healthcheck", "symptom_summary": "Health degraded"},
            {"incident_id": "inc-2", "repo": "AustinJR6/SoulsApp", "source": "github_actions", "symptom_summary": "Build failed"},
        ]
        controller.store.list_requested_investigations = lambda limit=10: incidents
        proposed = []
        controller.propose_fix = lambda incident: proposed.append(incident["incident_id"])

        controller.process_requested_investigations()

        self.assertEqual(proposed, ["inc-1", "inc-2"])

    def test_sync_resolved_workflow_incidents_marks_superseded_failures_deployed(self):
        controller = self._build_controller()
        incident = {
            "incident_id": "inc-1",
            "repo": "AustinJR6/SylVessel",
            "reproduction_hints": {
                "run_id": 100,
                "head_branch": "main",
                "workflow_name": "Deploy to DigitalOcean",
            },
            "metadata": {
                "workflow_run": {
                    "id": 100,
                    "path": ".github/workflows/deploy.yml",
                    "name": "Deploy to DigitalOcean",
                }
            },
        }
        controller.store.list_open_github_action_incidents = lambda limit=50: [incident]
        resolved = []
        controller.store.resolve_incident = lambda **kwargs: resolved.append(kwargs)
        controller.github.list_workflow_runs = lambda repo, branch=None, per_page=50: {
            "workflow_runs": [
                {
                    "id": 101,
                    "path": ".github/workflows/deploy.yml",
                    "conclusion": "success",
                    "html_url": "https://github.com/AustinJR6/SylVessel/actions/runs/101",
                }
            ]
        }

        controller.sync_resolved_workflow_incidents()

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["incident_id"], "inc-1")
        self.assertEqual(resolved[0]["status"], "deployed")
        self.assertIn("superseded", resolved[0]["root_cause_summary"])


if __name__ == "__main__":
    unittest.main()
