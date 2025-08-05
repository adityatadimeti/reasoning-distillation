# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python research project investigating methods for improving mathematical reasoning through summarization and reflection. The codebase focuses on distilling reasoning capabilities from large language models, particularly for mathematical problem-solving tasks (AIME, GPQA, GSM8K, HARP, Countdown).

## Key Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys:
# OPENAI_API_KEY=your-key
# FIREWORKS_API_KEY=your-key  
# TOGETHER_API_KEY=your-key
# ENABLE_API_CALLS=1
```

### Running Experiments
```bash
# Basic experiment run (config name only, no path needed)
python run_experiment.py <config_name>

# With web dashboard monitoring (port 8080 by default)
python run_experiment.py <config_name> --dashboard

# With parallel processing (N workers)
python run_experiment.py <config_name> --parallel --concurrency N

# Resume from previous reasoning results
python run_experiment.py <config_name> --load_initial_reasoning /path/to/previous_results.json

# Filter problems by ID or index
python run_experiment.py <config_name> --question_ids id1,id2,id3
python run_experiment.py <config_name> --exclude_question_ids id1,id2
python run_experiment.py <config_name> --index_range 0-4
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/experiments/test_reasoning_extraction.py

# Run with verbose output
pytest -v tests/
```

### Common Scripts
```bash
# Re-evaluate countdown results with updated validation
python scripts/reevaluate_countdown.py results_file.json

# View experiment results
python view_results.py results_file.json

# Countdown solver testing
python scripts/test_countdown_solver.py
```

## Architecture Overview

### Core Components

1. **LLM Clients** (`src/llm/`):
   - `openai_client.py`: OpenAI API integration
   - `fireworks_client.py`: Fireworks AI integration  
   - `together_client.py`: Together AI integration
   - `vllm_client.py`: Local vLLM server integration (see VLLM_GUIDE.md)
   - All clients implement async streaming with token tracking and cost estimation

2. **Experiment Types** (`src/experiments/`):
   - `summarization.py`: Main experiment runner for iterative reasoning + summarization
   - `passk.py`: Pass@K evaluation for measuring solution diversity
   - `pass_continuation.py`: Handles truncated solution continuations
   - Supports parallel processing, checkpointing, and real-time monitoring

3. **Reasoning Components** (`src/reasoning/`):
   - `extractor.py`: Answer extraction with dataset-specific parsers (math, countdown, multiple choice)
   - `summarizer.py`: Creates condensed summaries from reasoning traces
   - `HARP_utils.py`: Utilities for HARP dataset processing
   - Prompt templates configured via YAML in `config/prompts/`

4. **Evaluation** (`src/eval/`):
   - `countdown_check.py`: Specialized evaluation for Countdown number puzzles
   - `latex_answer_check.py`: LaTeX math expression evaluation
   - `parsing_lib.py`: General parsing utilities
   - Dataset-specific answer checking and validation

5. **Dashboard** (`src/dashboard/`):
   - Real-time web interface on port 8080 (configurable)
   - WebSocket-based updates for progress, token usage, costs
   - Auto-opens browser when using `--dashboard` flag

### Configuration System

Experiments use YAML configs in `config/experiments/` with these key sections:
- `experiment_type`: "summarize" (default) or "pass_k"
- `reasoning_model`: Model name/path and provider
- `summarizer_type`: "self" (same model) or specific model
- `prompts`: References to prompt versions in `config/prompts/`
- Generation parameters: temperature, max_tokens, top_p, etc.
- `answer_extractor`: Dataset-specific extractor ("default", "countdown", "mc")

### Data Processing Flow

1. **Input**: CSV files with columns: id, question, solution, answer
2. **Reasoning**: Model generates solution with configurable prompts
3. **Continuation**: Automatic handling of truncated responses
4. **Summarization**: Optional condensation of reasoning traces
5. **Extraction**: Dataset-specific answer parsing
6. **Evaluation**: Comparison against ground truth
7. **Output**: JSON results with full traces, metrics, and costs

### Key Implementation Details

- **Async Architecture**: All LLM calls use asyncio for concurrency
- **Streaming**: Token-by-token processing with finish reason tracking
- **Rate Limiting**: Built-in backoff and retry logic
- **Memory Efficiency**: Streaming prevents loading full responses in memory
- **Error Recovery**: Graceful handling of API failures and timeouts
- **Cost Tracking**: Real-time token counting and pricing calculations