#!/bin/zsh
#SBATCH --job-name=run_vllm_2
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops1
#SBATCH --output=slurm-output/serve_models.log
#SBATCH --error=slurm-output/serve_models.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

# Activate your conda or virtual environment
source /sailhome/jshen3/miniconda3/etc/profile.d/conda.sh
conda activate distill  # or use: conda activate myenv

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
# Launch Model 1 on GPU 0
# -----------------------------------
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8002 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > qwen2_2.log 2>&1 &

# -----------------------------------
# Launch Model 2 on GPU 1
# -----------------------------------
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --host 0.0.0.0 \
  --port 8003 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > deepseek_2.log 2>&1 &

# -----------------------------------
# Wait for both servers to be ready
# -----------------------------------
echo "Waiting for model servers to become available..."

until curl -s http://localhost:8002/v1/models &>/dev/null; do
  echo "Waiting for Qwen2.5 on port 8002..."
  sleep 5
done

until curl -s http://localhost:8003/v1/models &>/dev/null; do
  echo "Waiting for DeepSeek on port 8003..."
  sleep 5
done

echo "Both models are now available!"

# -----------------------------------
# Run experimental script (optional)
# -----------------------------------
echo "Running experiment script..."
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm_backtracking --parallel --concurrency 2 > experiment_2.log 2>&1 

