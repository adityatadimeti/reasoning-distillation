#!/bin/zsh
#SBATCH --job-name=qwen3
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops2
#SBATCH --output=slurm-output/serve_qwen3_8b.log
#SBATCH --error=slurm-output/serve_qwen3_8b.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Activate your conda or virtual environment
source /sailhome/jshen3/miniconda3/etc/profile.d/conda.sh
conda activate distill

export VLLM_USE_V1=0

# Hugging Face & transformers
export HF_HOME=/scr/jshen3/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# vLLM
export VLLM_CACHE_DIR=/scr/jshen3/vllm_cache

# pip cache
export PIP_CACHE_DIR=/scr/jshen3/pip_cache

# torch cache
export TORCH_COMPILE_CACHE=/scr/jshen3/torch_compile_cache

# -----------------------------------
# Launch Qwen3-14B model on GPU 0 (single model for both reasoning and summarization)
# -----------------------------------
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8005 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > qwen3_8b.log 2>&1 &

# -----------------------------------
# Wait for server to be ready
# -----------------------------------
echo "Waiting for Qwen3 model server to become available..."

until curl -s http://localhost:8005/v1/models &>/dev/null; do
  echo "Waiting for Qwen3 on port 8005..."
  sleep 5
done

echo "Qwen3 model is now available!"

# -----------------------------------
# Run experimental script (optional)
# -----------------------------------
echo "Running Qwen3 countdown experiment..."
python run_experiment.py countdown_qwen3_8b_vllm --parallel --concurrency 2 > qwen3_8b_experiment.log 2>&1