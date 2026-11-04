"""Línea de comandos: ``agroclima all`` reproduce el proyecto de punta a punta."""

from __future__ import annotations

import argparse
import logging
import sys

STAGES = ["download", "build", "inference", "forecast", "report"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agroclima",
        description="El Niño, La Niña y los cultivos de Colombia: análisis y alerta temprana.",
    )
    parser.add_argument(
        "etapas",
        nargs="+",
        choices=[*STAGES, "all"],
        help="Etapas a ejecutar, en orden. 'all' = " + " -> ".join(STAGES),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from .pipeline import run

    stages = STAGES if "all" in args.etapas else [s for s in STAGES if s in args.etapas]
    run(stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
