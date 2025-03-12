#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    source .env
fi

# Default values
CONFIG_FILE="configs/experiments/self_summarization.yaml"
METHOD="external"
MAX_ITERATIONS=2
SUMMARIZATION_MODE="append"
PROBLEM_IDS="2024-I-8"  # Test with single problem
OUTPUT_DIR="results/model_sweep_test"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --summarization-mode)
            SUMMARIZATION_MODE="$2"
            shift 2
            ;;
        --problem-ids)
            PROBLEM_IDS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run sweep for each model
echo "Running model sweep test with configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Method: $METHOD"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Summarization mode: $SUMMARIZATION_MODE"
echo "  Problem IDs: $PROBLEM_IDS"
echo "  Output directory: $OUTPUT_DIR"

# List of model configs to test
MODEL_CONFIGS=(
    "configs/models/deepseek_r1.yaml"
    "configs/models/qwen_7b.yaml"
    "configs/models/llama_8b.yaml"
    "configs/models/qwen_1p5b.yaml"
)

# Run each model config
for model_config in "${MODEL_CONFIGS[@]}"; do
    echo -e "\nRunning sweep with config: $model_config"
    echo "----------------------------------------"
    
    ./scripts/run_summarization_sweep.sh \
        --config "$model_config" \
        --method "$METHOD" \
        --max-iterations "$MAX_ITERATIONS" \
        --summarization-mode "$SUMMARIZATION_MODE" \
        --problem-ids "$PROBLEM_IDS"
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Successfully completed sweep for $model_config"
    else
        echo "Error running sweep for $model_config"
    fi
done

# Run self-summarization as baseline
echo -e "\nRunning self-summarization baseline"
echo "----------------------------------------"
./scripts/run_summarization_sweep.sh \
    --config "$CONFIG_FILE" \
    --method "self" \
    --max-iterations "$MAX_ITERATIONS" \
    --summarization-mode "$SUMMARIZATION_MODE" \
    --problem-ids "$PROBLEM_IDS"

echo -e "\nModel sweep test completed" 