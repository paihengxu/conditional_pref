# Periodic Automation & Slurm Bridge Utilities

This directory and `slurm_bridge/` contain production-ready utilities to run continuous coding agent iterations and execute Slurm workloads across a login/compute node boundary.

---

## 1. Periodic Automation (`auto_task/`)

The `auto_task/` framework allows you to run periodic coding agent passes in the background to continuously execute evaluations, explore parameters, or heal the codebase.

### Directory Layout
```text
auto_task/                  # Periodic automation agent orchestration
├── README.md               # This documentation file
├── run_codex_monitor.sh    # Shell wrapper executing the automation pass
├── cron_example.txt        # Crontab line template for scheduling the automation
├── automation_task.md      # Master instructions read by the agent
├── automation_state.md     # Running state log appended to by the agent
├── minimal_agent_task.md   # Diagnostic minimal agent instruction file
└── logs/                   # Saved prompts, raw execution logs, and agent final outputs
    └── .gitkeep
```

- **Workflow**: A cron job calls `run_codex_monitor.sh`. It compiles instructions from `automation_task.md` and the history in `automation_state.md`, then launches an autonomous workspace execution pass (using `codex` or another CLI agent).
- **Setup**:
  1. Customize `auto_task/automation_task.md` with your system guidelines and target objectives.
  2. Test a run manually using `bash auto_task/run_codex_monitor.sh`.
  3. Schedule periodic runs using the crontab pattern illustrated in `auto_task/cron_example.txt`.

---

## 2. Slurm Bridge (`slurm_bridge/`)

ML projects are often developed on compute nodes that lack `sbatch` permission or direct login access to the scheduler. The `slurm_bridge/` provides a secure, minimal, shared-filesystem queueing mechanism.

### Directory Layout
```text
slurm_bridge/               # Shared-filesystem control-action bridge for Slurm
├── login_worker.py         # Runs on login node to poll and execute Slurm requests
├── write_request.py        # Runs on compute node to submit Slurm requests
├── status_snapshot.py      # Runs on login node to generate squeue/sacct snapshot JSON
├── README.md               # Detailed documentation for the Slurm bridge
├── sample_crontab.txt      # Sample crontab lines for the login_worker
├── sample_status_crontab.txt # Sample crontab lines for the status snapshot
├── requests/               # Queued request files (.json)
│   └── .gitkeep
├── in_progress/            # Requests currently being handled
│   └── .gitkeep
├── done/                   # Handled and archived requests
│   └── .gitkeep
├── responses/              # Output responses from the worker (.json)
│   └── .gitkeep
├── status/                 # Output status snapshots (.json)
│   └── .gitkeep
└── logs/                   # Cron logs and diagnostic output files
    └── .gitkeep
```

- **Workflow**:
  1. **Compute node**: Write an sbatch script and call `write_request.py submit` to drop a request.
  2. **Login node**: A cron job periodically triggers `login_worker.py --once`. It processes pending submissions, queries, or cancellations, executing actual scheduler commands (`sbatch`, `squeue`, `sacct`, `scancel`) and saving results.
  3. **Compute node**: Read responses in `slurm_bridge/responses/` and current queue state in `slurm_bridge/status/latest.json` (refreshed by `status_snapshot.py` on the login node).
- **Setup**: See `slurm_bridge/README.md` for detailed specifications and scheduling configuration.
