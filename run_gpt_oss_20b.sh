#!/bin/zsh
#SBATCH --job-name=gpt-oss-20b
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops-hgx-1
#SBATCH --output=slurm-output/serve_gpt_oss_20b.log
#SBATCH --error=slurm-output/serve_gpt_oss_20b.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=36:00:00

# Activate your conda or virtual environment
source /sailhome/jshen3/miniconda3/etc/profile.d/conda.sh
conda activate distill

export VLLM_USE_V1=0

# Hugging Face & transformers
export HF_HOME=/scr/jshen3/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
# export HF_TOKEN="your_token_here"

# vLLM
export VLLM_CACHE_DIR=/scr/jshen3/vllm_cache

# pip cache
export PIP_CACHE_DIR=/scr/jshen3/pip_cache

# torch cache
export TORCH_COMPILE_CACHE=/scr/jshen3/torch_compile_cache

# -----------------------------------
# Launch GPT-OSS-20B for reasoning on GPU 0 (high effort)
# -----------------------------------
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8012 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > gpt_oss_reasoning.log 2>&1 &

# -----------------------------------
# Launch GPT-OSS-20B for summarization on GPU 1 (low effort)
# -----------------------------------
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8013 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > gpt_oss_summarization.log 2>&1 &

# -----------------------------------
# Wait for both servers to be ready
# -----------------------------------
echo "Waiting for GPT-OSS-20B model servers to become available..."

until curl -s http://localhost:8012/v1/models &>/dev/null; do
  echo "Waiting for GPT-OSS-20B reasoning server on port 8012..."
  sleep 5
done

until curl -s http://localhost:8013/v1/models &>/dev/null; do
  echo "Waiting for GPT-OSS-20B summarization server on port 8013..."
  sleep 5
done

echo "Both GPT-OSS-20B reasoning and summarization servers are now available!"

# -----------------------------------
# Run experimental script (optional)
# -----------------------------------
echo "Running GPT-OSS-20B countdown experiment..."
python run_experiment.py countdown_gpt_oss_20b_vllm --parallel --concurrency 2 > gpt_oss_experiment.log 2>&1