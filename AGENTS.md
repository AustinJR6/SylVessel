Sylana Vessel workspace agent rules:

- FastAPI backend in `server.py` is the orchestration source of truth.
- Prefer existing memory, scheduler, and work-session primitives before inventing parallel systems.
- Isolated sessions should not pollute the main conversation unless they explicitly announce back.
- Keep outputs practical and execution-focused.
