#!/bin/bash

# Run recursive refinement experiment

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export EXPERIMENT_NAME="recursive_refinement"

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/reasoning_traces
mkdir -p data/summaries
mkdir -p results/recursive_pipeline
mkdir -p logs

# Run the experiment
echo "Running recursive refinement experiment..."
python experiments/run_experiment.py --config configs/experiments/recursive_refinement.yaml

echo "Experiment complete. Results saved to results/recursive_pipeline/"
