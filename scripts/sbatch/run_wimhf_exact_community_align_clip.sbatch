#!/bin/bash
#SBATCH --job-name=wimhf_exact_full_community_align_gpt5mini_s42_clip_20260608
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=2-00:00:00
#SBATCH --qos=medium
#SBATCH --mem=64GB
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --output=outputs/slurm/wimhf_exact_full_community_align_gpt5mini_s42_clip_20260608_%j.log

source /nfshomes/paiheng/.bashrc
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
conda activate condpref

export PYTHONUNBUFFERED=1

if [[ -z "${OAI_WIMHF:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export OAI_WIMHF="${OPENAI_API_KEY}"
fi

python scripts/run_wimhf.py \
  --config configs/community_align_wimhf_exact.json \
  --mode full \
  --run-root "outputs/reproduction/wimhf_exact/wimhf_exact_full_community_align_gpt5mini_s42_clip_20260608_${SLURM_JOB_ID:-manual}"

conda deactivate
