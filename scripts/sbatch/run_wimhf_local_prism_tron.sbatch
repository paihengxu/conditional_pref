#!/bin/bash
#SBATCH --job-name=wimhf_local_full_prism_s42_tron_20260607
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --time=1-00:00:00
#SBATCH --qos=high
#SBATCH --mem=64GB
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --output=outputs/reproduction/wimhf_local/wimhf_local_full_prism_s42_tron_20260607_%j.log

source /nfshomes/paiheng/.bashrc
cd /fs/clip-projects/clip-k12/paiheng/conditional_pref
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
conda activate textdiff

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WIMHF_VLLM_TENSOR_PARALLEL_SIZE=4
export WIMHF_VLLM_MAX_MODEL_LEN=8192
export WIMHF_VLLM_GPU_MEMORY_UTILIZATION=0.85

if [[ -z "${OAI_WIMHF:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export OAI_WIMHF="${OPENAI_API_KEY}"
fi

python scripts/run_wimhf.py prism \
  --mode full \
  --run-root "outputs/reproduction/wimhf_local/wimhf_local_full_prism_s42_tron_20260607_${SLURM_JOB_ID:-manual}" \
  --local-model Qwen/Qwen3-30B-A3B-Instruct-2507

conda deactivate
