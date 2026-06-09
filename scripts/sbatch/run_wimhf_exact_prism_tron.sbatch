#!/bin/bash
#SBATCH --job-name=wimhf_exact_full_prism_gpt5mini_s42_tron_20260608
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=1-00:00:00
#SBATCH --qos=high
#SBATCH --mem=64GB
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --output=outputs/slurm/wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_%j.log

source /nfshomes/paiheng/.bashrc
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
conda activate textdiff

export PYTHONUNBUFFERED=1

if [[ -z "${OAI_WIMHF:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export OAI_WIMHF="${OPENAI_API_KEY}"
fi

python scripts/run_wimhf.py prism \
  --mode full \
  --run-root "outputs/reproduction/wimhf_exact/wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_${SLURM_JOB_ID:-manual}" \
  --annotator-model gpt-5-mini

conda deactivate
