# Reasoning Distillation

This project investigates methods for improving mathematical reasoning through summarization and reflection.

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/adityatadimeti/reasoning-distillation.git
   cd reasoning-distillation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys:
   
   Create a `.env` file in the root directory with your API keys:
   ```
   # API Keys
   OPENAI_API_KEY=your_openai_api_key
   FIREWORKS_API_KEY=your_fireworks_api_key
   
   # Enable API calls
   ENABLE_API_CALLS=1
   ```

## Configuration

### Experiment Configuration

Experiments are configured using YAML files in the `config/experiments` directory. A typical configuration file looks like:

```yaml
# Experiment configuration
experiment_name: "test_reasoning_problems"
results_dir: "./results/test"
data_path: "./data/simple.csv"
save_intermediate: true

# Dashboard configuration
dashboard_port: 8080

# Model configuration
reasoning_model: "accounts/fireworks/models/qwq-32b"
summarizer_type: "self"

# Generation parameters
max_tokens: 16384
temperature: 0.7
top_p: 1.0
top_k: 40
presence_penalty: 0.0
frequency_penalty: 0.0

# Summarization parameters
enable_summarization: true
summary_max_tokens: 16384
summary_temperature: 0.7
summary_top_p: 1.0
summary_top_k: 40
summary_presence_penalty: 0.0
summary_frequency_penalty: 0.0

# Prompt configurations
prompts:
  reasoning: "v0"  # This references the version in config/prompts/reasoning.yaml
  summarize: "testing"  # This references the version in config/prompts/summarize.yaml
```

### Problem Sets

Problem sets are defined in CSV files in the `data` directory:

```csv
id,question,solution,answer
simple,What is 5 + 3?,5 + 3=8,8
```

Each row contains:
- `id`: Unique identifier for the problem
- `question`: The question text
- `solution`: The solution or explanation
- `answer`: The expected answer

## Running Experiments

To run an experiment, use the `run_experiment.py` script:

```bash
python run_experiment.py <config_name> [options]
```

### Reusing Initial Reasoning

You can reuse initial reasoning from a previous experiment run to test different summarization prompts while keeping the initial reasoning constant:

```bash
python run_experiment.py <config_name> --load_initial_reasoning=/path/to/previous_results.json
```

This is useful when:
1. You want to experiment with different summarization prompt templates
2. You want to test different summarization models
3. You want to try different summarization parameters

By reusing initial reasoning:
- All "iteration 0" solutions stay the same
- Only the summarization steps and subsequent iterations change
- You save cost and time by not re-generating the initial reasoning

### Other Options

- `--dashboard`: Enable the web dashboard for real-time experiment monitoring
- `--verbose`: Log detailed information about LLM calls
- `--parallel`: Process problems in parallel (incompatible with dashboard)
- `--concurrency N`: Set maximum number of parallel problems (default: 4) 
- `--question_ids`: Comma-separated list of problem IDs to process
- `--exclude_question_ids`: Comma-separated list of problem IDs to exclude
- `--index_range`: Range of indices to process (e.g., "0-4" or "10-15")

## Project Structure

- `run_experiment.py`: Main entry point for running experiments
- `src/`:
  - `experiments/`: Contains experiment implementations
    - `summarization.py`: Summarization experiment implementation
  - `llm/`: LLM client implementations
    - `openai_client.py`: OpenAI API client
    - `fireworks_client.py`: Fireworks AI API client
  - `reasoning/`: Reasoning and summarization utilities
    - `extractor.py`: Extracts answers from reasoning texts
    - `summarizer.py`: Summarizes reasoning traces
  - `dashboard/`: Dashboard for experiment visualization
- `config/`:
  - `experiments/`: Experiment configuration files
  - `prompts/`: Prompt templates
- `data/`: Problem sets in CSV format
- `results/`: Output directory for experiment results

## Results

Experiment results are saved in the specified `results_dir` directory from your configuration. The results include:

- `config.json`: The configuration used for the experiment
- `results.json`: The results of the experiment, including questions, answers, and evaluations

## Tips

1. When working with Fireworks models, always include the `top_k` parameter in your configuration.
2. Use the `--verbose` flag during development to see the exact prompts and parameters sent to the LLM.
3. The dashboard (`--dashboard` flag) provides real-time feedback on experiment progress.
4. Make sure `ENABLE_API_CALLS=1` is set in your `.env` file to allow the system to make actual API calls.
5. Check the logs for detailed information about the experiment run.
