#!/bin/zsh
#SBATCH --job-name=ds_14b_backtracking
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops2
#SBATCH --output=slurm-output/serve_ds_14b_backtracking.log
#SBATCH --error=slurm-output/serve_ds_14b_backtracking.err
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
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --host 0.0.0.0 \
  --port 8017 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > ds_14b_backtracking_reasoning.log 2>&1 &

# -----------------------------------
# Launch Qwen3-14B for summarization on GPU 1
# -----------------------------------
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8018 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > ds_14b_backtracking_summarization.log 2>&1 &

# -----------------------------------
# Wait for both servers to be ready
# -----------------------------------
echo "Waiting for DeepSeek-R1-Distill-Qwen-14B model servers to become available..."

until curl -s http://localhost:8017/v1/models &>/dev/null; do
  echo "Waiting for DeepSeek-R1-Distill-Qwen-14B reasoning server on port 8017..."
  sleep 5
done

until curl -s http://localhost:8018/v1/models &>/dev/null; do
  echo "Waiting for Qwen2.5 summarization server on port 8018..."
  sleep 5
done

echo "Both DeepSeek-R1-Distill-Qwen-14B and Qwen2.5 models are now available!"

# -----------------------------------
# Run experimental script (optional)
# -----------------------------------
echo "Running DeepSeek-R1-Distill-Qwen-14B and Qwen2.5 countdown experiment..."
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm_backtracking --parallel --concurrency 2 > ds_14b_backtracking_experiment.log 2>&1