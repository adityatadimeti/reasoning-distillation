# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python research project investigating methods for improving mathematical reasoning through summarization and reflection. The codebase focuses on distilling reasoning capabilities from large language models, particularly for mathematical problem-solving tasks (AIME, GPQA, GSM8K, HARP).

## Key Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys:
# OPENAI_API_KEY=your-key
# FIREWORKS_API_KEY=your-key  
# TOGETHER_API_KEY=your-key
```

### Running Experiments
```bash
# Basic experiment run
python run_experiment.py <config_name>

# With web dashboard monitoring
python run_experiment.py <config_name> --dashboard

# With parallel processing (N workers)
python run_experiment.py <config_name> --parallel --concurrency N

# Resume from previous reasoning results
python run_experiment.py <config_name> --load_initial_reasoning

# Run batch experiments
python -m scripts.batch_run_experiments
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_reasoning_extraction.py

# Run with verbose output
pytest -v tests/
```

### Development Scripts
```bash
# Test LLM connections
python scripts/test_llm.py
python scripts/test_vllm.py

# Analyze results
python scripts/analyze_results.py
python scripts/model_size_analysis.py

# Data processing
python scripts/convert_to_csv.py
```

## Architecture Overview

### Core Components

1. **LLM Clients** (`src/llm/`):
   - `openai_client.py`: OpenAI API integration
   - `fireworks_client.py`: Fireworks AI integration
   - `together_client.py`: Together AI integration
   - `vllm_client.py`: Local vLLM server integration
   - All clients implement async streaming interface with token tracking

2. **Experiment Framework** (`src/experiments/`):
   - `summarization_experiment.py`: Main experiment runner for reasoning + summarization
   - `pass_k_experiment.py`: Pass@K evaluation experiments
   - Handles iteration loops, continuation for long traces, parallel processing

3. **Reasoning Engine** (`src/reasoning/`):
   - `extraction.py`: Answer extraction from mathematical solutions
   - `summarization.py`: Creates summaries from reasoning traces
   - Supports multiple prompt templates configurable via YAML

4. **Dashboard** (`src/dashboard/`):
   - Real-time web interface for monitoring experiments
   - Shows progress, token usage, costs, and results
   - Built with asyncio and websockets

### Configuration System

- Experiments configured via YAML files in `config/experiments/`
- Prompt templates in `config/prompts/`
- Supports model-specific settings, temperature, max tokens, etc.
- Can override settings via command-line arguments

### Data Flow

1. Problems loaded from CSV files in `data/` directory
2. Reasoning phase: LLM generates solution traces
3. Optional summarization: Creates condensed versions of reasoning
4. Answer extraction: Parses mathematical answers from text
5. Results saved to `results/` directory as JSON

### Key Design Patterns

- **Async everywhere**: Uses asyncio for concurrent API calls
- **Streaming responses**: Handles token-by-token streaming from LLMs
- **Continuation support**: Automatically continues when hitting token limits
- **Checkpoint/resume**: Can save and load intermediate results
- **Cost tracking**: Monitors API usage and estimates costs

## Important Considerations

1. **Token Limits**: Different models have different context windows. The system handles continuation automatically but be aware of limits.

2. **API Keys**: Required in `.env` file. The system will fail gracefully if keys are missing.

3. **vLLM Setup**: For local inference, requires separate vLLM server. See VLLM_GUIDE.md for setup instructions.

4. **Parallel Processing**: Use `--concurrency` flag carefully. Too many parallel requests can hit rate limits.

5. **Results Storage**: All results saved in `results/` with timestamps. Check disk space for large experiments.

6. **Answer Extraction**: The system uses sophisticated regex patterns for mathematical answers. Check `src/reasoning/extraction.py` for supported formats.

## Common Development Tasks

### Adding a New LLM Provider
1. Create new client in `src/llm/` implementing the base interface
2. Add provider to `src/llm/factory.py`
3. Update experiment configs to use new provider

### Modifying Prompts
1. Edit YAML files in `config/prompts/`
2. Reference new prompts in experiment configs
3. Test with small dataset first

### Adding New Datasets
1. Convert to CSV format with columns: id, question, solution, answer
2. Place in `data/` directory
3. Update experiment configs to reference new dataset

### Debugging Experiments
1. Use `--verbose` flag for detailed logging
2. Check `results/` for intermediate outputs
3. Use dashboard (`--dashboard`) for real-time monitoring
4. Small test runs: limit problems in config file