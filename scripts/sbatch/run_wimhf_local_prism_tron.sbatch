#!/bin/bash
#SBATCH --job-name=wimhf_local_full_prism_qwen35_27b_s42_clip_20260609
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=1-00:00:00
#SBATCH --qos=medium
#SBATCH --mem=64GB
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --output=outputs/reproduction/wimhf_local/wimhf_local_full_prism_qwen35_27b_s42_clip_20260609_%j.log

source /nfshomes/paiheng/.bashrc
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
conda activate textdiff

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export WIMHF_VLLM_TENSOR_PARALLEL_SIZE=2
export WIMHF_VLLM_MAX_MODEL_LEN=8192
export WIMHF_VLLM_GPU_MEMORY_UTILIZATION=0.85

if [[ -z "${OAI_WIMHF:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export OAI_WIMHF="${OPENAI_API_KEY}"
fi

python scripts/run_wimhf.py \
  --config configs/prism_wimhf_local.json \
  --mode full \
  --run-root "outputs/reproduction/wimhf_local/wimhf_local_full_prism_qwen35_27b_s42_clip_20260609_${SLURM_JOB_ID:-manual}"

conda deactivate
