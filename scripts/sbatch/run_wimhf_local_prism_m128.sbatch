#!/bin/bash
#SBATCH --job-name=wimhf_local_prism_qwen35_m128_s42_clip_20260610
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=1-00:00:00
#SBATCH --qos=medium
#SBATCH --mem=64GB
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --output=outputs/slurm/wimhf_local_prism_qwen35_m128_s42_clip_20260610_%j.log

set -e

source /nfshomes/paiheng/.bashrc
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
module load cuda/13.0.2
conda activate condpref

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export WIMHF_VLLM_TENSOR_PARALLEL_SIZE=2
export WIMHF_VLLM_MAX_MODEL_LEN=8192
export WIMHF_VLLM_GPU_MEMORY_UTILIZATION=0.85

if [[ -z "${OAI_WIMHF:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export OAI_WIMHF="${OPENAI_API_KEY}"
fi

RUN_ROOT="outputs/reproduction/wimhf_local/wimhf_local_full_prism_qwen35_27b_m128_s42_clip_20260610_${SLURM_JOB_ID:-manual}"

python scripts/run_wimhf.py \
  --config configs/prism_wimhf_local_m128.json \
  --mode full \
  --run-root "${RUN_ROOT}"

python scripts/run_prism_covariate_selector.py \
  --run-root "${RUN_ROOT}" \
  --covariate-column conversation_type \
  --method demeaned-reweighted-lasso \
  --top-k 5 10

echo "WIMHF_RUN_ROOT=${RUN_ROOT}"
conda deactivate
