# vLLM Integration Guide for Reasoning Distillation

## Overview

This guide explains how to use vLLM (Versatile Large Language Model) for local inference with the reasoning distillation codebase. vLLM provides high-throughput, memory-efficient inference for large language models.

## What is vLLM?

vLLM is an open-source library for fast LLM inference and serving. Key features:
- **High throughput**: 2-32x higher than HuggingFace Transformers
- **PagedAttention**: Efficient memory management
- **Continuous batching**: Maximizes GPU utilization
- **OpenAI-compatible API**: Easy integration with existing code
- **Support for many models**: Llama, Qwen, DeepSeek, Mistral, and more

## Installation

1. **Install vLLM** (requires CUDA-capable GPU):
```bash
pip install vllm
```

2. **Install additional dependencies**:
```bash
pip install aiohttp requests
```

## Setting Up vLLM Server

### Basic Server Start

To start a vLLM server with a model:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

### Advanced Configuration

For better performance and memory management:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --max-num-seqs 256
```

Parameters explained:
- `--max-model-len`: Maximum context length (default: model's max)
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.9 = 90%)
- `--dtype`: Data type (auto, float16, bfloat16, float32)
- `--max-num-seqs`: Maximum number of sequences to process

### Multi-GPU Setup

For models that require multiple GPUs:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

## Using vLLM with the Codebase

### 1. Configuration

The codebase now supports vLLM through configuration files. Example:

```yaml
# Model configuration for vLLM
reasoning_model: "Qwen/Qwen2.5-14B-Instruct"
reasoning_model_provider: "vllm"

# vLLM server configuration
vllm_config:
  host: "localhost"
  port: 8000
  max_model_len: 32768
```

### 2. Running Experiments

With vLLM server running, execute experiments:

```bash
# Run a single experiment
python run_experiment.py aime_deepseek_qwen_14b_summ_base_sum_4iter_vllm

# With dashboard for monitoring
python run_experiment.py aime_deepseek_qwen_14b_summ_base_sum_4iter_vllm --dashboard

# Run in parallel for faster processing
python run_experiment.py aime_deepseek_qwen_14b_summ_base_sum_4iter_vllm --parallel --concurrency 8
```

### 3. Multiple Model Servers

To use different models for reasoning and summarization:

1. Start first vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000
```

2. Start second vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8001
```

3. Configure in YAML:
```yaml
reasoning_model: "Qwen/Qwen2.5-14B-Instruct"
reasoning_model_provider: "vllm"
vllm_config:
  host: "localhost"
  port: 8000

summarizer_model: "Qwen/Qwen2.5-7B-Instruct"
summarizer_model_provider: "vllm"
# Note: You'd need to modify the code to support different vLLM configs for summarizer
```

## Supported Models

vLLM supports many models. For reasoning tasks, recommended models include:

- **Qwen Series**: `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`
- **DeepSeek**: Models available on HuggingFace
- **Llama**: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.3`

## Performance Optimization

### 1. Batch Processing

Enable parallel processing for multiple problems:
```bash
python run_experiment.py config_name --parallel --concurrency 16
```

### 2. GPU Memory Management

Monitor GPU usage and adjust:
```bash
nvidia-smi -l 1  # Monitor every second
```

Adjust `--gpu-memory-utilization` if OOM errors occur.

### 3. Context Length

For long reasoning traces, ensure `max_model_len` is sufficient:
- 32K tokens: Good for most AIME problems
- 64K+ tokens: Needed for complex multi-step reasoning

## Troubleshooting

### Common Issues

1. **Server not reachable**:
   - Check firewall settings
   - Ensure server is running
   - Verify host/port in config

2. **Out of Memory**:
   - Reduce `--gpu-memory-utilization`
   - Use smaller model
   - Reduce `--max-num-seqs`

3. **Slow inference**:
   - Check GPU utilization
   - Increase batch size
   - Use faster GPU

### Debug Mode

Enable verbose logging:
```bash
python run_experiment.py config_name --verbose
```

Check vLLM server logs for detailed information.

## Cost Comparison

Using vLLM vs API services:

| Service | Cost per 1M tokens | 1000 AIME problems (~50M tokens) |
|---------|-------------------|----------------------------------|
| Fireworks API | $0.90 | $45.00 |
| OpenAI GPT-4 | $30.00 | $1,500.00 |
| vLLM (local) | $0.00* | $0.00* |

*Electricity and hardware costs not included

## Advanced Features

### 1. Custom Sampling

Modify generation parameters in config:
```yaml
temperature: 0.6
top_p: 0.95
top_k: 40
presence_penalty: 0.0
frequency_penalty: 0.0
```

### 2. Streaming

The vLLM client supports streaming for real-time output (not used by default in experiments).

### 3. Health Monitoring

Check server health:
```bash
curl http://localhost:8000/health
```

Get model info:
```bash
curl http://localhost:8000/v1/models
```

## Best Practices

1. **Model Selection**: Choose models based on your GPU memory
   - 24GB GPU: Up to 14B parameter models
   - 48GB GPU: Up to 32B parameter models
   - 80GB GPU: Up to 70B parameter models

2. **Server Management**: Use process managers like `supervisord` for production

3. **Monitoring**: Set up logging and metrics collection

4. **Backup**: Keep Fireworks API as fallback for critical runs

## Example Workflow

1. Start vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct
```

2. Test connection:
```bash
curl http://localhost:8000/health
```

3. Run experiment:
```bash
python run_experiment.py aime_deepseek_qwen_14b_summ_base_sum_4iter_vllm --dashboard
```

4. Monitor progress at http://localhost:8080 (dashboard)

5. Check results in `./results/` directory

## Migration from Fireworks

To migrate existing experiments:

1. Copy the config file: `cp config.yaml config_vllm.yaml`
2. Update model provider: `reasoning_model_provider: "vllm"`
3. Add vLLM config section
4. Update model names to HuggingFace format
5. Run with new config

## Support

For issues specific to:
- **vLLM**: Check [vLLM GitHub](https://github.com/vllm-project/vllm)
- **This codebase**: Check the error logs and traceback
- **Model compatibility**: Verify model support in vLLM docs 