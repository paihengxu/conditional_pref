# Slurm Job Launching

This repo can launch Slurm jobs in two ways:

1. Directly from a login node with `sbatch`; `ssh nexusclip00` first.
2. Ask a human to run the submission command for you.

## Slurm Preference

Follow this structure for generated or handwritten sbatch scripts:

```bash
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --nodes=1
#SBATCH --gres=gpu:<gpu_type>:<count>
#SBATCH --time=<limit>
#SBATCH --qos=<qos>
#SBATCH --mem=<memory>
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --output=<log_dir>/<job_name>.log

source /nfshomes/paiheng/.bashrc
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
conda activate textdiff

python <runner>.py <args>

conda deactivate
```

Reusable conventions:

- Use `/nfshomes/paiheng/.bashrc` before `conda activate`.
- `cd` into the repo before running Python.
- Use the `textdiff` conda environment (for now).
- Put one command per sbatch script.
- Give every job a descriptive `--job-name`.
- Write logs to an explicit experiment log directory rather than Slurm defaults.
- Include method, dataset setting, seed, and date or sweep name in generated job names.

## Working Partition Configs

The common default is the CLIP partition:

```bash
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=2-00:00:00
#SBATCH --qos=medium
#SBATCH --mem=64GB
#SBATCH --account=clip
#SBATCH --partition=clip
```

When there's two jobs at CLIP, use TRON:

```bash
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --time=1-00:00:00
#SBATCH --qos=high
#SBATCH --mem=64GB
#SBATCH --account=nexus
#SBATCH --partition=tron
```

Time can be adjusted based on the compute job.

## Generated Scripts

`../text_diff/textDiff/utils/sweep_utils.py` has a reusable `create_sbatch_script()` helper. Its defaults are:

```python
nodes=1
gpus="rtxa6000:2"
time="8:30:00"
qos="medium"
mem="64GB"
account="clip"
partition="clip"
conda_env="textdiff"
working_dir="/fs/clip-projects/clip-k12/paiheng/text_diff"
```

The helper emits the same basic script shape:

```bash
source /nfshomes/paiheng/.bashrc
cd <working_dir>
conda activate <conda_env>

<cmd_line>

conda deactivate
```

When adapting this pattern here, set:

```text
working_dir="/fs/clip-projects/clip-k12/paiheng/conditional_pref"
```

## Submission Pattern

From the login node:

```bash
ssh nexusclip00
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
sbatch scripts/sbatch/<job>.sbatch
```

For sweeps, generate scripts first, then submit each script with `sbatch`.
Use names that map Slurm jobs back to datasets, seeds, methods, and logs without opening the script.

## Logs

Use separate log roots for different job families:

```text
outputs/reproduction/<job_family>/<job_name>.log
```

Use a more specific directory when running a sweep.

## Runner Command Pattern

The command line inside each sbatch script should be explicit and reproducible.


## Checklist

Before submitting:

- Confirm you are on `nexusclip00` or ask a human to submit.
- Use a descriptive `--job-name`.
- Use the right partition/account/qos tuple.
- Source bashrc, enter the repo, and activate `textdiff`.
- Write sbatch logs to a unique path.
- Put experiment outputs under a run-specific directory.
- Keep one seed or one dataset shard per job when that improves rerun and cancellation granularity.

After submitting:

- Record the Slurm job id.
- Check the sbatch log before assuming missing metrics mean the job never ran.
- If rerunning, use a new job/log name so stale logs cannot be mistaken for fresh output.
