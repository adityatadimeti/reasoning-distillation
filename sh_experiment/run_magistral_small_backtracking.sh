#!/bin/zsh
#SBATCH --job-name=mm_small_backtracking
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops-hgx-1
#SBATCH --output=slurm-output/serve_mm_small_backtracking.log
#SBATCH --error=slurm-output/serve_mm_small_backtracking.err
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
# Launch Magistral Small 2506 for reasoning on GPU 0
# -----------------------------------
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Magistral-Small-2506 \
  --tokenizer mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8002 \
  --max-model-len 40960 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > magistral_small_backtracking_reasoning.log 2>&1 &

# -----------------------------------
# Launch Mistral Small 3.1 for summarization on GPU 1
# -----------------------------------
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8003 \
  --max-model-len 40960 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > mistral_small_backtracking_summarization.log 2>&1 &

# -----------------------------------
# Wait for both servers to be ready
# -----------------------------------
echo "Waiting for Magistral and Mistral model servers to become available..."

until curl -s http://localhost:8002/v1/models &>/dev/null; do
  echo "Waiting for Magistral reasoning server on port 8002..."
  sleep 5
done

until curl -s http://localhost:8003/v1/models &>/dev/null; do
  echo "Waiting for Mistral summarization server on port 8003..."
  sleep 5
done

echo "Both Magistral and Mistral models are now available!"

# -----------------------------------
# Run experimental script (optional)
# -----------------------------------
echo "Running Magistral/Mistral countdown experiment..."
python run_experiment.py countdown_magistral_mistral_vllm_backtracking --parallel --concurrency 2 > magistral_small_backtracking_experiment.log 2>&1