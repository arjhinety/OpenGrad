#!/usr/bin/env python
"""The authoritative publication-readiness gate.

Run this before declaring a release, report or publication complete. It is the
release interface; the `network`-marked tests exist only to catch regressions in
the machinery, not to replace this command.

The output always includes an execution census, because a gate that exits 0
without having examined anything is indistinguishable from a gate that examined
everything and found nothing wrong. Both happened here: the freeze gate skipped
every canonical corpus, and `python -m opengrad.registry.validate` ran no checks
at all and exited 0.

Exit codes:
    0   PASS    -- every required population was examined and passed
    1   FAIL    -- a demonstrable provenance defect exists
    2   BLOCKED -- a required check could not run; nothing is verified

A blocked check is never reported as success. If the network is unavailable or an
optional dependency is missing, this command says so and exits non-zero, because
"we could not check" and "we checked and it is fine" are different claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # Imported here rather than at module scope so the script also runs straight
    # from a checkout that has not been installed, and so the import block above
    # stays a single sorted block of stdlib imports.
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from opengrad.publication import (
        BLOCKED_NETWORK,
        BLOCKED_OPTIONAL_DEPENDENCY,
        FAIL,
        PASS,
        verify_publication,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip network resolution; the affected pins are reported BLOCKED, never dropped",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="per-request timeout in seconds"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the report as machine-readable JSON"
    )
    args = parser.parse_args(argv)

    report = verify_publication(Path(args.root), network=not args.offline, timeout=args.timeout)

    if args.json:
        print(
            json.dumps(
                {
                    "verifier_contract": report.contract,
                    "overall": report.overall,
                    "totals": report.totals(),
                    "gates": [
                        {
                            "name": result.name,
                            "status": result.status,
                            "policy": result.policy,
                            "precondition": result.precondition,
                            **result.counts(),
                            "detail": result.detail,
                            "skipped": [
                                {"id": skip.id, "reason": skip.reason} for skip in result.skipped
                            ],
                            "errors": result.all_errors(),
                        }
                        for result in report.results
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(report.render())

    if report.overall == PASS:
        return 0
    if report.overall == FAIL:
        return 1
    if report.overall == BLOCKED_OPTIONAL_DEPENDENCY:
        print("\ninstall the missing optional dependency and re-run", file=sys.stderr)
    if report.overall == BLOCKED_NETWORK:
        print(
            "\nno network access, so upstream pins were not resolved; "
            "re-run with connectivity before declaring this verified",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
