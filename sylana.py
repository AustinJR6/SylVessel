"""Legacy compatibility entrypoint for Sylana CLI."""

from entrypoints.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
