#!/bin/bash
# Script to run summarization sweep on existing reasoning traces

# Set the Python path to include the project root
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found"
fi

# Verify API key is set
if [ -z "$FIREWORKS_API_KEY" ]; then
    echo "Error: FIREWORKS_API_KEY environment variable not set"
    echo "Please create a .env file with your API key or set it in your environment"
    exit 1
fi

# Default values
CONFIG=""
METHOD="self"
MAX_ITERATIONS=2
PROBLEM_IDS=""
SUMMARIZATION_MODE="append"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            CONFIG="$2"
            shift
            shift
            ;;
        --method)
            METHOD="$2"
            shift
            shift
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift
            shift
            ;;
        --problem-ids)
            PROBLEM_IDS="$2"
            shift
            shift
            ;;
        --summarization-mode)
            SUMMARIZATION_MODE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default config based on method if not specified
if [ -z "$CONFIG" ]; then
    if [ "$METHOD" == "self" ]; then
        CONFIG="configs/experiments/self_summarization.yaml"
    else
        CONFIG="configs/experiments/external_summarization.yaml"
    fi
fi

echo "Running summarization sweep with configuration:"
echo "  Config file: $CONFIG"
echo "  Method: $METHOD"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Summarization mode: $SUMMARIZATION_MODE"
if [ -n "$PROBLEM_IDS" ]; then
    echo "  Problem IDs: $PROBLEM_IDS"
fi

# Create output directories if they don't exist
mkdir -p results/summarization_sweep
mkdir -p logs

# Enable API calls for this run
export ENABLE_API_CALLS=1

# Run the summarization sweep
python experiments/summarization_sweep.py \
    --config "$CONFIG" \
    --method "$METHOD" \
    --max-iterations "$MAX_ITERATIONS" \
    --summarization-mode "$SUMMARIZATION_MODE" \
    ${PROBLEM_IDS:+--problem-ids "$PROBLEM_IDS"}

# Check if the experiment was successful
if [ $? -eq 0 ]; then
    echo "Summarization sweep completed successfully"
else
    echo "Summarization sweep failed"
    exit 1
fi 