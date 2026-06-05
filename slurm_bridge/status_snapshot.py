#!/usr/bin/env python3
"""Write a Slurm status snapshot for the coding agent to read later."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
STATUS_DIR = ROOT / "status"

SQUEUE_FIELDS = [
    "job_id",
    "state",
    "elapsed",
    "reason",
    "job_name",
    "partition",
    "submit_time",
    "start_time",
    "nodes",
]

SACCT_FIELDS = [
    "JobID",
    "JobName",
    "Partition",
    "State",
    "Elapsed",
    "Start",
    "End",
    "ExitCode",
    "NodeList",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_start_time(hours_back: int) -> str:
    return (utc_now() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")


def run_command(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def parse_squeue(stdout: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not stdout:
        return rows
    for line in stdout.splitlines():
        parts = line.split("|")
        if len(parts) != len(SQUEUE_FIELDS):
            rows.append({"raw": line})
            continue
        rows.append(dict(zip(SQUEUE_FIELDS, parts)))
    return rows


def parse_sacct(stdout: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not stdout:
        return rows

    reader = csv.reader(io.StringIO(stdout), delimiter="|")
    for record in reader:
        if not record:
            continue
        if len(record) != len(SACCT_FIELDS):
            rows.append({"raw": "|".join(record)})
            continue
        entry = dict(zip(SACCT_FIELDS, record))
        job_id = entry.get("JobID", "")
        if "." in job_id:
            continue
        rows.append(entry)
        if len(rows) >= limit:
            break
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user",
        default=os.environ.get("USER", ""),
        help="Slurm user to snapshot. Defaults to $USER.",
    )
    parser.add_argument(
        "--output",
        default=str(STATUS_DIR / "latest.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=24,
        help="How far back sacct should look for recent jobs.",
    )
    parser.add_argument(
        "--recent-limit",
        type=int,
        default=200,
        help="Maximum number of top-level sacct rows to retain.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    user = str(args.user).strip()
    if not user:
        parser.error("--user is required when $USER is empty")

    output_path = Path(args.output).expanduser()

    squeue = run_command(
        [
            "squeue",
            "--user",
            user,
            "--noheader",
            "--format",
            "%.18i|%.9T|%.10M|%.20R|%.50j|%.12P|%.19V|%.19S|%.6D",
        ]
    )
    sacct = run_command(
        [
            "sacct",
            "--user",
            user,
            "--parsable2",
            "--noheader",
            "--starttime",
            format_start_time(args.hours_back),
            "--format",
            ",".join(SACCT_FIELDS),
        ]
    )

    payload = {
        "generated_at": utc_now().isoformat(),
        "user": user,
        "snapshot_ok": squeue["returncode"] == 0 and sacct["returncode"] == 0,
        "active_jobs": parse_squeue(squeue["stdout"]),
        "recent_jobs": parse_sacct(sacct["stdout"], args.recent_limit),
        "squeue": {
            "command": squeue["command"],
            "returncode": squeue["returncode"],
            "stderr": squeue["stderr"],
        },
        "sacct": {
            "command": sacct["command"],
            "returncode": sacct["returncode"],
            "stderr": sacct["stderr"],
            "hours_back": args.hours_back,
        },
    }
    write_json(output_path, payload)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
