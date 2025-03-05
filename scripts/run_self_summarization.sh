#!/bin/bash

# Run self-summarization experiment

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export EXPERIMENT_NAME="self_summarization"

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/reasoning_traces
mkdir -p data/summaries
mkdir -p results/self_summarization
mkdir -p logs

# Run the experiment
echo "Running self-summarization experiment..."
python experiments/run_experiment.py --config configs/experiments/self_summarization.yaml

echo "Experiment complete. Results saved to results/self_summarization/"
