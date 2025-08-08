#!/bin/zsh
#SBATCH --job-name=magistral-mistral
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops-hgx-1
#SBATCH --output=slurm-output/serve_magistral_mistral.log
#SBATCH --error=slurm-output/serve_magistral_mistral.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
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
# Launch Magistral Small 2506 for reasoning on GPU 0
# -----------------------------------
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Magistral-Small-2506 \
  --host 0.0.0.0 \
  --port 8010 \
  --max-model-len 40960 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > magistral_reasoning.log 2>&1 &

# -----------------------------------
# Launch Mistral Small 3.1 for summarization on GPU 1
# -----------------------------------
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --host 0.0.0.0 \
  --port 8011 \
  --max-model-len 40960 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > mistral_summarization.log 2>&1 &

# -----------------------------------
# Wait for both servers to be ready
# -----------------------------------
echo "Waiting for Magistral and Mistral model servers to become available..."

until curl -s http://localhost:8010/v1/models &>/dev/null; do
  echo "Waiting for Magistral reasoning server on port 8010..."
  sleep 5
done

until curl -s http://localhost:8011/v1/models &>/dev/null; do
  echo "Waiting for Mistral summarization server on port 8011..."
  sleep 5
done

echo "Both Magistral and Mistral models are now available!"

# -----------------------------------
# Run experimental script (optional)
# -----------------------------------
echo "Running Magistral/Mistral countdown experiment..."
python run_experiment.py countdown_magistral_mistral_vllm --parallel --concurrency 2 > magistral_mistral_experiment.log 2>&1