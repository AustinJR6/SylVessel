"""Legacy compatibility entrypoint for quantized Sylana CLI."""

from entrypoints.quantized import main


if __name__ == "__main__":
    raise SystemExit(main())
