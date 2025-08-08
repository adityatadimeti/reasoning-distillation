#!/bin/zsh
#SBATCH --job-name=run_qwen3_vllm
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops1
#SBATCH --output=slurm-output/serve_qwen3_models.log
#SBATCH --error=slurm-output/serve_qwen3_models.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1  # Only need one GPU for single Qwen3-14B model
#SBATCH --time=36:00:00

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
 --model Qwen/Qwen3-14B \
 --host 0.0.0.0 \
 --port 8000 \
 --max-model-len 32768 \
 --dtype bfloat16 \
 --gpu-memory-utilization 0.85 \
 > qwen3.log 2>&1 &

# -----------------------------------
# Wait for server to be ready
# -----------------------------------
echo "Waiting for Qwen3 model server to become available..."

until curl -s http://localhost:8000/v1/models &>/dev/null; do
 echo "Waiting for Qwen3 on port 8000..."
 sleep 5
done

echo "Qwen3 model is now available!"

# -----------------------------------
# Run experimental script (optional)
# -----------------------------------
echo "Running Qwen3 countdown experiment..."
python run_experiment.py countdown_qwen3_14b_vllm --parallel --concurrency 2 --load_initial_reasoning /afs/cs.stanford.edu/u/jshen3/research_projects/reasoning-distillation/results/countdown_deepseek_rl_qwen2_5_vllm/countdown_deepseek_rl_qwen2_5_vllm_20250729_035159/results.json > qwen3_experiment.log 2>&1