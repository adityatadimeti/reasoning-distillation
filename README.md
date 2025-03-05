# Reasoning Distillation Research Project

## Project Structure

```
reasoning-enhancement/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation
├── .gitignore              # Git ignore patterns
├── .env.example            # Example environment variables
├── configs/                # Configuration files
├── src/                    # Source code
├── scripts/                # Utility scripts
├── experiments/            # Experiment runners
├── notebooks/              # Analysis notebooks
├── data/                   # Data storage
├── results/                # Experiment results
├── logs/                   # Experiment logs
└── tests/                  # Unit tests
```

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables: `cp .env.example .env` and edit as needed
4. Run a baseline experiment: `./scripts/run_baseline.sh`