#!/bin/bash
# Script to run the baseline reasoning experiment

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
CONFIG="configs/experiments/baseline.yaml"
SPLIT="test"
MAX_PROBLEMS=1  # Limit for quicker testing

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            CONFIG="$2"
            shift
            shift
            ;;
        --split)
            SPLIT="$2"
            shift
            shift
            ;;
        --max-problems)
            MAX_PROBLEMS="$2"
            shift
            shift
            ;;
        --problem-ids)
            PROBLEM_IDS="$2"
            shift
            shift
            ;;
        --all)
            MAX_PROBLEMS=""  # Process all problems
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running baseline experiment with configuration:"
echo "  Config file: $CONFIG"
echo "  Data split: $SPLIT"
if [ -n "$MAX_PROBLEMS" ]; then
    echo "  Max problems: $MAX_PROBLEMS"
else
    echo "  Processing all problems"
fi
if [ -n "$PROBLEM_IDS" ]; then
    echo "  Problem IDs: $PROBLEM_IDS"
fi

# Create output directories if they don't exist
mkdir -p results
mkdir -p logs

# Enable API tests for this run
export ENABLE_API_CALLS=1
export ENABLE_GPT4O_EXTRACTION=1

# Run the experiment
python experiments/baseline.py \
    --config "$CONFIG" \
    --split "$SPLIT" \
    ${MAX_PROBLEMS:+--max-problems "$MAX_PROBLEMS"} \
    ${PROBLEM_IDS:+--problem-ids "$PROBLEM_IDS"}

# Check if the experiment was successful
if [ $? -eq 0 ]; then
    echo "Baseline experiment completed successfully"
else
    echo "Baseline experiment failed"
    exit 1
fi