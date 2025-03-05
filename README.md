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

## GPT-4o Answer Extraction

For complex mathematical problems (like AIME), the standard regex-based answer extraction may not be sufficient. The project includes a GPT-4o-based extraction fallback that can be enabled with environment variables:

1. Set `ENABLE_GPT4O_EXTRACTION=1` to enable GPT-4o extraction
2. Set `OPENAI_API_KEY=your_api_key` with your OpenAI API key

When enabled, the system will:
1. First try standard regex-based extraction methods
2. If those fail, fall back to GPT-4o to extract the answer

To run tests with GPT-4o extraction:

```bash
ENABLE_GPT4O_EXTRACTION=1 OPENAI_API_KEY=your_api_key pytest tests/test_reasoning.py::TestReasoningGeneration::test_generate_with_aime_problem -v
```

Note: Using GPT-4o extraction will incur OpenAI API costs.