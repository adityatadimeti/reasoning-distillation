#!/bin/zsh
#SBATCH --job-name=q3_14b_base_summ
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops2
#SBATCH --output=slurm-output/serve_q3_14b_base_summ.log
#SBATCH --error=slurm-output/serve_q3_14b_base_summ.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00

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
# Launch Qwen3-14B for reasoning on GPU 0
# -----------------------------------
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8007 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  > qwen3_base_summ_reasoning.log 2>&1 &

# -----------------------------------
# Launch Qwen3-14B for summarization on GPU 1
# -----------------------------------
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8008 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  > qwen3_base_summ_summarization.log 2>&1 &

# -----------------------------------
# Wait for both servers to be ready
# -----------------------------------
echo "Waiting for Qwen3 model servers to become available..."

until curl -s http://localhost:8007/v1/models &>/dev/null; do
  echo "Waiting for Qwen3 reasoning server on port 8007..."
  sleep 5
done

until curl -s http://localhost:8008/v1/models &>/dev/null; do
  echo "Waiting for Qwen3 summarization server on port 8008..."
  sleep 5
done

echo "Both Qwen3 models are now available!"

# -----------------------------------
# Run experimental script (optional)
# -----------------------------------
echo "Running Qwen3 countdown experiment..."
python run_experiment.py countdown_qwen3_14b_vllm --parallel --concurrency 2 > qwen3_14b_base_summ_experiment.log 2>&1