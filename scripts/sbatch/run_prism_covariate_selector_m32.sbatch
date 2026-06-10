#!/bin/bash
#SBATCH --job-name=prism_covariate_selector_m32_s42_tron_20260610
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=04:00:00
#SBATCH --qos=high
#SBATCH --mem=64GB
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --output=outputs/slurm/prism_covariate_selector_m32_s42_tron_20260610_%j.log

source /nfshomes/paiheng/.bashrc
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
conda activate condpref

export PYTHONUNBUFFERED=1
WIMHF_RUN_ROOT="${WIMHF_RUN_ROOT:-outputs/reproduction/wimhf_local/wimhf_local_full_prism_qwen35_27b_s42_clip_20260609_6986752}"

python scripts/run_prism_covariate_selector.py \
  --run-root "${WIMHF_RUN_ROOT}" \
  --covariate-column conversation_type \
  --method demeaned-reweighted-lasso \
  --top-k 5 10

conda deactivate
