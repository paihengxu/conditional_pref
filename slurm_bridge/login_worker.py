#!/usr/bin/env python3
"""Process queued Slurm control requests from a shared filesystem."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REQUESTS_DIR = ROOT / "requests"
IN_PROGRESS_DIR = ROOT / "in_progress"
DONE_DIR = ROOT / "done"
RESPONSES_DIR = ROOT / "responses"
LOGS_DIR = ROOT / "logs"

QUEUE_DIRS = (
    REQUESTS_DIR,
    IN_PROGRESS_DIR,
    DONE_DIR,
    RESPONSES_DIR,
    LOGS_DIR,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in QUEUE_DIRS:
        path.mkdir(parents=True, exist_ok=True)


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


def parse_job_id(stdout: str) -> str | None:
    parts = stdout.strip().split()
    if len(parts) >= 4 and parts[-1].isdigit():
        return parts[-1]
    return None


def read_request(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def validate_request(payload: dict[str, Any]) -> tuple[str, str]:
    request_id = str(payload.get("request_id", "")).strip()
    request_type = str(payload.get("type", "")).strip()
    if not request_id:
        raise ValueError("missing request_id")
    if not request_type:
        raise ValueError("missing type")
    return request_id, request_type


def handle_submit(payload: dict[str, Any]) -> dict[str, Any]:
    sbatch_script = Path(str(payload.get("sbatch_script", ""))).expanduser()
    if not sbatch_script.is_file():
        raise ValueError(f"sbatch_script not found: {sbatch_script}")
    result = run_command(["sbatch", str(sbatch_script)])
    response: dict[str, Any] = {
        "ok": result["returncode"] == 0,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }
    job_id = parse_job_id(result["stdout"])
    if job_id is not None:
        response["job_id"] = job_id
    return response


def handle_status(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(payload.get("job_id", "")).strip()
    if not job_id:
        raise ValueError("missing job_id")
    squeue = run_command(
        ["squeue", "--jobs", job_id, "--noheader", "--format", "%.18i|%.9T|%.10M|%.20R|%.20j"]
    )
    sacct = run_command(
        ["sacct", "-j", job_id, "--parsable2", "--noheader", "--format", "JobID,State,Elapsed,ExitCode"]
    )
    return {
        "ok": squeue["returncode"] == 0 and sacct["returncode"] == 0,
        "squeue": {
            "stdout": squeue["stdout"],
            "stderr": squeue["stderr"],
            "returncode": squeue["returncode"],
        },
        "sacct": {
            "stdout": sacct["stdout"],
            "stderr": sacct["stderr"],
            "returncode": sacct["returncode"],
        },
    }


def handle_cancel(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(payload.get("job_id", "")).strip()
    if not job_id:
        raise ValueError("missing job_id")
    result = run_command(["scancel", job_id])
    return {
        "ok": result["returncode"] == 0,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def handle_tail_log(payload: dict[str, Any]) -> dict[str, Any]:
    log_path = Path(str(payload.get("log_path", ""))).expanduser()
    lines = int(payload.get("lines", 40))
    if lines <= 0:
        raise ValueError("lines must be positive")
    if not log_path.is_file():
        raise ValueError(f"log_path not found: {log_path}")
    result = run_command(["tail", "-n", str(lines), str(log_path)])
    return {
        "ok": result["returncode"] == 0,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


HANDLERS = {
    "submit": handle_submit,
    "status": handle_status,
    "cancel": handle_cancel,
    "tail_log": handle_tail_log,
}


def process_request(request_path: Path) -> dict[str, Any]:
    payload = read_request(request_path)
    request_id, request_type = validate_request(payload)
    handler = HANDLERS.get(request_type)
    if handler is None:
        raise ValueError(f"unsupported request type: {request_type}")
    response = handler(payload)
    response["request_id"] = request_id
    response["type"] = request_type
    response["processed_at"] = utc_now()
    return response


def archive_request(source: Path, request_id: str) -> None:
    archived_path = DONE_DIR / f"{request_id}.json"
    if archived_path.exists():
        archived_path.unlink()
    shutil.move(str(source), str(archived_path))


def process_one(request_path: Path) -> None:
    in_progress_path = IN_PROGRESS_DIR / request_path.name
    try:
        request_path.replace(in_progress_path)
    except FileNotFoundError:
        return

    request_id = request_path.stem
    try:
        response = process_request(in_progress_path)
        request_id = str(response.get("request_id", request_id))
    except Exception as exc:  # noqa: BLE001
        response = {
            "request_id": request_id,
            "ok": False,
            "error": str(exc),
            "processed_at": utc_now(),
        }

    response_path = RESPONSES_DIR / f"{request_id}.json"
    write_json(response_path, response)
    archive_request(in_progress_path, request_id)


def process_queue() -> int:
    ensure_dirs()
    processed = 0
    for request_path in sorted(REQUESTS_DIR.glob("*.json")):
        process_one(request_path)
        processed += 1
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="process the current queue once and exit",
    )
    args = parser.parse_args()

    processed = process_queue()
    if args.once:
        print(f"processed {processed} request(s)")
        return 0

    print("This worker is intended to be run with --once from cron.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
