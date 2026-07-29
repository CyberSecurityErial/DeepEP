#!/usr/bin/env python3
"""Emit C105 rail-balance validation bundles.

This runner is intentionally validation-only.  It prepares deterministic route,
traffic, and CPU-oracle plan evidence for real-cluster runs, but it never marks
Hybrid Rail/Gin runtime correctness as passed.
"""

import argparse
import json
from pathlib import Path

from rail_balance_validation_common import (
    CANONICAL_CASES,
    RESULT_SCHEMA_VERSION,
    build_validation_bundle,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--case",
        choices=("all",) + CANONICAL_CASES,
        default="all",
    )
    parser.add_argument("--mode", choices=("off", "force", "both"), default="both")
    parser.add_argument("--num-scaleout-ranks", type=int, required=True)
    parser.add_argument("--num-scaleup-ranks", type=int, required=True)
    parser.add_argument("--num-tokens-per-rank", type=int, required=True)
    parser.add_argument("--num-topk", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--num-channels", type=int, required=True)
    parser.add_argument("--proxy-slots-per-rank", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = CANONICAL_CASES if args.case == "all" else (args.case,)
    modes = ("off", "force") if args.mode == "both" else (args.mode,)
    bundles = [
        build_validation_bundle(
            run_id=f"{args.run_id}/{case}",
            case=case,
            modes=modes,
            num_scaleout_ranks=args.num_scaleout_ranks,
            num_scaleup_ranks=args.num_scaleup_ranks,
            num_tokens_per_rank=args.num_tokens_per_rank,
            num_topk=args.num_topk,
            num_experts=args.num_experts,
            hidden=args.hidden,
            num_channels=args.num_channels,
            proxy_slots_per_rank=args.proxy_slots_per_rank,
        )
        for case in cases
    ]
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundles": bundles,
    }
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
