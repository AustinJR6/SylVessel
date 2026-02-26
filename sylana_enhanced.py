"""Legacy compatibility entrypoint for enhanced Sylana CLI."""

from entrypoints.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
