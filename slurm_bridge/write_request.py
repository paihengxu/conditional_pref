#!/usr/bin/env python3
"""Write Slurm bridge request files from the compute node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REQUESTS_DIR = ROOT / "requests"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="type", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--request-id", required=True)
    submit.add_argument("--sbatch-script", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--request-id", required=True)
    status.add_argument("--job-id", required=True)

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--request-id", required=True)
    cancel.add_argument("--job-id", required=True)

    tail_log = subparsers.add_parser("tail_log")
    tail_log.add_argument("--request-id", required=True)
    tail_log.add_argument("--log-path", required=True)
    tail_log.add_argument("--lines", type=int, default=40)

    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": args.request_id,
        "type": args.type,
    }
    if args.type == "submit":
        payload["sbatch_script"] = args.sbatch_script
    elif args.type in {"status", "cancel"}:
        payload["job_id"] = args.job_id
    elif args.type == "tail_log":
        payload["log_path"] = args.log_path
        payload["lines"] = args.lines
    return payload


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    request_path = REQUESTS_DIR / f"{args.request_id}.json"
    write_json(request_path, payload)
    print(request_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
