#!/bin/bash

# Run baseline experiment

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export EXPERIMENT_NAME="baseline"

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/reasoning_traces
mkdir -p results/baseline
mkdir -p logs

# Run the experiment
echo "Running baseline experiment..."
python experiments/run_experiment.py --config configs/experiments/baseline.yaml

echo "Experiment complete. Results saved to results/baseline/"
