# Maintenance Controller

The maintenance controller is a separate approval-gated repair service for Vessel and Lysara. It should run on its own small host, not on the main Vessel droplet.

## What it does

- watches production health and failed GitHub Actions runs
- records deduped repair incidents in Supabase
- opens isolated repo clones for bounded repair attempts
- runs `codex exec` with a structured repair-output schema
- opens GitHub PRs instead of pushing directly to `main`
- creates approval items in the Vessel review center
- marks repair runs as merged and deployed after GitHub confirms the downstream workflow succeeded

## Safety defaults

- never pushes directly to `main`
- never edits secrets
- never performs destructive database changes
- stops at diagnosis when an incident is ambiguous or infra-related
- app-repo repairs trigger preview builds only

## Environment

Copy [`.env.maintenance.template`](../.env.maintenance.template) onto the maintenance host and set:

- `SUPABASE_DB_URL`
- `GITHUB_TOKEN`
- `VESSEL_API_BASE_URL`
- `MAINTENANCE_READ_TOKEN`
- `MAINTENANCE_REPAIR_MODE`

The Vessel backend must also have the same `MAINTENANCE_READ_TOKEN` configured so `/repairs/logs/tail` can be read safely.
Recommended first setting:

- `MAINTENANCE_REPAIR_MODE=diagnosis`

## Usage

Run a single polling pass:

```bash
python -m maintenance_controller.controller run-once
```

Run the continuous loop:

```bash
python -m maintenance_controller.controller loop
```

## Required external setup

- Branch protection on `main`
- GitHub Actions enabled on the backend and app repos
- `EXPO_TOKEN` secret present in the app repo for repair preview builds
- A host or VM separate from the Vessel droplet for this controller
- Backend repo secrets for maintenance droplet deploy:
  - `MAINTENANCE_DO_HOST`
  - `MAINTENANCE_DO_USERNAME`
  - `MAINTENANCE_DO_SSH_KEY`
- Backend repo variable:
  - `MAINTENANCE_DEPLOY_ENABLED=true`
