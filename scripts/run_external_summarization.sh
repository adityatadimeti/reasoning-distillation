#!/bin/bash

# Run external summarization experiment

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export EXPERIMENT_NAME="external_summarization"

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/reasoning_traces
mkdir -p data/summaries
mkdir -p results/external_summarization
mkdir -p logs

# Run the experiment
echo "Running external summarization experiment..."
python experiments/run_experiment.py --config configs/experiments/external_summarization.yaml

echo "Experiment complete. Results saved to results/external_summarization/"
