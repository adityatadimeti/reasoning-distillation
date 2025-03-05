#!/bin/bash

# Setup environment for experiments

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e .
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/reasoning_traces
mkdir -p data/summaries
mkdir -p results
mkdir -p logs

# Setup environment variables
echo "Setting up environment variables..."
cp .env.example .env
echo "Please edit .env file with your API keys and configuration."

echo "Environment setup complete."
