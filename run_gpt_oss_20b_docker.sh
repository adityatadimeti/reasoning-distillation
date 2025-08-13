#!/bin/bash
#SBATCH --job-name=gpt-oss-20b-docker
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops-hgx-1
#SBATCH --output=slurm-output/serve_gpt_oss_20b_docker.log
#SBATCH --error=slurm-output/serve_gpt_oss_20b_docker.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=36:00:00

# Hugging Face token for model access


echo "Starting GPT-OSS-20B Docker containers..."

# -----------------------------------
# Stop and remove any existing containers
# -----------------------------------
echo "Cleaning up existing containers..."
docker stop gpt-oss-reasoning gpt-oss-summarization 2>/dev/null || true
docker rm gpt-oss-reasoning gpt-oss-summarization 2>/dev/null || true

# -----------------------------------
# Launch GPT-OSS-20B reasoning server on GPU 0
# -----------------------------------
echo "Starting GPT-OSS reasoning server on GPU 0, port 8012..."
docker run -d \
    --name gpt-oss-reasoning \
    --gpus '"device=0"' \
    -p 8012:8000 \
    --ipc=host \
    -e HF_TOKEN="$HF_TOKEN" \
    vllm/vllm-openai:gptoss \
    --model openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85

# -----------------------------------
# Launch GPT-OSS-20B summarization server on GPU 1
# -----------------------------------
echo "Starting GPT-OSS summarization server on GPU 1, port 8013..."
docker run -d \
    --name gpt-oss-summarization \
    --gpus '"device=1"' \
    -p 8013:8000 \
    --ipc=host \
    -e HF_TOKEN="$HF_TOKEN" \
    vllm/vllm-openai:gptoss \
    --model openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85

# -----------------------------------
# Wait for both servers to be ready
# -----------------------------------
echo "Waiting for GPT-OSS Docker containers to be ready..."

# Wait for reasoning server
echo "Checking reasoning server on port 8012..."
until curl -s http://localhost:8012/v1/models &>/dev/null; do
    echo "Waiting for reasoning server..."
    sleep 10
done
echo "✓ Reasoning server is ready"

# Wait for summarization server  
echo "Checking summarization server on port 8013..."
until curl -s http://localhost:8013/v1/models &>/dev/null; do
    echo "Waiting for summarization server..."
    sleep 10
done
echo "✓ Summarization server is ready"

echo "Both GPT-OSS Docker containers are running!"

# Show container status
echo "Container status:"
docker ps | grep gpt-oss

# Show logs (last 10 lines)
echo -e "\nReasoning server logs:"
docker logs --tail 10 gpt-oss-reasoning

echo -e "\nSummarization server logs:"
docker logs --tail 10 gpt-oss-summarization

echo -e "\nServers ready! You can now run experiments."
echo "Reasoning server: http://localhost:8012"
echo "Summarization server: http://localhost:8013"

# Optional: Run experiment automatically
# echo "Running GPT-OSS experiment..."
# conda activate distill
# python run_experiment.py countdown_gpt_oss_20b_vllm --parallel --concurrency 2
