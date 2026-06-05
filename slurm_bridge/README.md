# Slurm Bridge

This directory contains a minimal shared-filesystem bridge for running Slurm
control actions on a login node while keeping experiment code and analysis on a
compute node.

## Model

1. A compute-node session writes JSON request files into `requests/`.
2. A lightweight login-node worker processes those requests.
3. The worker writes JSON responses into `responses/`.
4. Logs and experiment outputs stay in their normal shared filesystem paths.

The login-node worker should only perform lightweight control-plane actions:

- `sbatch`
- `squeue`
- `sacct`
- `scancel`
- brief `tail` for logs

It should not run training, evaluation, or heavy parsing.

## Layout

Expected directories:

```text
slurm_bridge/
  requests/
  in_progress/
  done/
  responses/
  status/
  logs/
```

These are included with `.gitkeep` files so the worker can run immediately.

## Request Format

Each request is a single JSON file in `requests/` with a unique `request_id`
and a `type`.

### Submit

```json
{
  "request_id": "submit-exp-001",
  "type": "submit",
  "sbatch_script": "/shared/project/jobs/exp_001.sbatch"
}
```

### Status

```json
{
  "request_id": "status-exp-001",
  "type": "status",
  "job_id": "8123456"
}
```

### Cancel

```json
{
  "request_id": "cancel-exp-001",
  "type": "cancel",
  "job_id": "8123456"
}
```

### Tail Log

```json
{
  "request_id": "tail-exp-001",
  "type": "tail_log",
  "log_path": "/shared/project/logs/8123456.out",
  "lines": 40
}
```

## Running The Worker

Run this on the login node:

```bash
python slurm_bridge/login_worker.py --once
```

Or from cron on whatever latency you want for request processing. For example, to run every minute:

```bash
* * * * * cd /path/to/your/project && /usr/bin/bash -lc 'source ~/.bashrc && conda activate <your_env> && python slurm_bridge/login_worker.py --once' >> /path/to/your/project/slurm_bridge/logs/cron.log 2>&1
```

See `sample_crontab.txt` for a copyable example.

## Writing A Status Snapshot

For the coding agent, it is often better to refresh a standalone snapshot right
before the agent starts rather than polling continuously through the worker.

Run this on the login node:

```bash
python slurm_bridge/status_snapshot.py --user <your_username> --output slurm_bridge/status/latest.json
```

The snapshot includes:

- `active_jobs`: current `squeue` rows for the user
- `recent_jobs`: recent top-level `sacct` rows for the user
- `generated_at`: snapshot timestamp

If the coding session runs every 12 hours, schedule the snapshot 10 minutes
before the agent run. See `sample_status_crontab.txt` for a copyable example.

## Writing Requests

From the compute node:

```bash
python slurm_bridge/write_request.py submit --request-id submit-exp-001 --sbatch-script /shared/project/jobs/exp_001.sbatch
python slurm_bridge/write_request.py status --request-id status-exp-001 --job-id 8123456
```

## Responses

Responses are written to `responses/<request_id>.json`.

Example:

```json
{
  "request_id": "submit-exp-001",
  "ok": true,
  "type": "submit",
  "job_id": "8123456",
  "stdout": "Submitted batch job 8123456"
}
```
