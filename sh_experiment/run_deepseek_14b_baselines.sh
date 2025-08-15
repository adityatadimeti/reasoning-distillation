#!/bin/zsh
#SBATCH --job-name=ds_14b_baselines
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops1
#SBATCH --output=slurm-output/serve_deepseek_14b_baselines.log
#SBATCH --error=slurm-output/serve_deepseek_14b_baselines.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
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
  --port 8019 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > deepseek_14b_baselines_reasoning.log 2>&1 &


# -----------------------------------
# Wait for both servers to be ready
# -----------------------------------
echo "Waiting for DeepSeek-R1-Distill-Qwen-14B model servers to become available..."

until curl -s http://localhost:8019/v1/models &>/dev/null; do
  echo "Waiting for DeepSeek-R1-Distill-Qwen-14B reasoning server on port 8019..."
  sleep 5
done


echo "DeepSeek-R1-Distill-Qwen-14B model is now available!"

# -----------------------------------
# Run experimental script (optional)
# -----------------------------------
echo "Running DeepSeek-R1-Distill-Qwen-14B baselines experiment..."
python run_experiment.py countdown_deepseek_rl_vllm_baselines --parallel --concurrency 2 > deepseek_14b_baselines_experiment.log 2>&1