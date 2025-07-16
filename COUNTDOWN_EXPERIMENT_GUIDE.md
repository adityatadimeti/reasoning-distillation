# Comprehensive Guide: Running Countdown Experiments

This guide walks you through the complete process of running Countdown number puzzle experiments using the reasoning distillation codebase with your vLLM setup.

## 📋 **Prerequisites**

### 1. **vLLM Servers Running**
Ensure your SLURM script is running both models:
```bash
# Check if servers are active
curl http://localhost:8000/v1/models  # Qwen2.5-14B-Instruct
curl http://localhost:8001/v1/models  # DeepSeek-RL-Distill-Qwen-14B
```

### 2. **Environment Setup**
```bash
# Activate your conda environment
conda activate distill

# Ensure you're in the project directory
cd /path/to/reasoning-distillation
```

## 🗂️ **Step 1: Generate the Countdown Dataset**

### Generate CSV with Solutions
```bash
# This will download the dataset and generate solutions
python scripts/convert_countdown_to_csv.py
```

**What this does:**
- Downloads Countdown dataset from HuggingFace
- Generates solutions using ALL numbers exactly once
- Creates `data/countdown.csv` with columns:
  - `id`: Problem identifier
  - `question`: Full prompt with `<answer>` tag instructions
  - `solution`: Mathematical expression (e.g., `"(25 + 7) * 2 - 3"`)
  - `answer`: Target number
  - `human_eval`: Step-by-step explanation

**Expected output:**
```
Loading Countdown dataset from Huggingface...
Loaded 1000 problems
Generating solutions for countdown problems...
Processed 100/1000 problems...
Processed 200/1000 problems...
...
Converted 1000 problems and saved to data/countdown.csv
```

## 🧪 **Step 2: Run Countdown Experiments**

### Basic Experiment Run
```bash
# Run basic experiment
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm
```

### With Real-Time Dashboard
```bash
# Run with web dashboard for monitoring
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm --dashboard
```
Then open http://localhost:8080 in your browser to monitor progress.

### Parallel Processing (Recommended)
```bash
# Run with parallel processing for faster execution
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm --parallel --concurrency 8
```

### Test Run on Subset
```bash
# Test on first 10 problems
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm --index_range "0-9" --parallel --concurrency 8

# Test on specific problems
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm --question_ids "countdown_1,countdown_5,countdown_10"
```

## 📊 **Step 3: Monitor and Analyze Results**

### Real-Time Monitoring
If using `--dashboard`, you can monitor:
- **Progress**: Number of problems completed
- **Token Usage**: Input/output tokens per model
- **Cost Tracking**: Estimated costs (will be $0 for vLLM)
- **Success Rate**: How many problems are being solved correctly
- **Current Problem**: What the model is currently working on

### Results Location
Results are saved to: `./results/countdown_deepseek_rl_qwen2_5_vllm/`

**Key files:**
- `results.json`: Main results file with all responses
- `config.yaml`: Copy of experiment configuration
- `summary.json`: Aggregate statistics

### Check Progress
```bash
# Check how many problems completed
ls ./results/countdown_deepseek_rl_qwen2_5_vllm/

# Quick view of results
head -20 ./results/countdown_deepseek_rl_qwen2_5_vllm/results.json
```

## 🔧 **Step 4: Advanced Experiment Options**

### Resume Interrupted Experiments
```bash
# If experiment was interrupted, it will automatically resume
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm
```

### Use Previous Reasoning Results
```bash
# Reuse reasoning from previous run, only generate new summaries
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm --load_initial_reasoning
```

### Verbose Logging
```bash
# Enable detailed logging of all LLM calls
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm --verbose
```

### Custom Concurrency
```bash
# Adjust based on your GPU memory and compute
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm --parallel --concurrency 4   # Conservative
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm --parallel --concurrency 16  # Aggressive
```

## 📈 **Step 5: Analyze Experiment Results**

### Basic Statistics
```bash
# Run analysis script (if available)
python scripts/analyze_results.py ./results/countdown_deepseek_rl_qwen2_5_vllm/results.json
```

### Custom Analysis
Create your own analysis script:
```python
import json
import pandas as pd

# Load results
with open('./results/countdown_deepseek_rl_qwen2_5_vllm/results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)

# Calculate success rate
success_rate = df['correct'].mean() if 'correct' in df.columns else 0
print(f"Success rate: {success_rate:.2%}")

# Analyze by iteration
for iteration in range(4):
    iter_correct = df[f'iteration_{iteration}_correct'].mean()
    print(f"Iteration {iteration} success rate: {iter_correct:.2%}")
```

## 🛠️ **Step 6: Experiment Customization**

### Modify Experiment Parameters
Edit `config/experiments/countdown_deepseek_rl_qwen2_5_vllm.yaml`:

```yaml
# Adjust iteration count
max_iterations: 2  # Reduce for faster experiments

# Modify token limits
max_tokens: 4096   # Reduce for simpler problems
max_total_tokens: 16384

# Change temperature for more/less randomness
temperature: 0.3   # More deterministic
temperature: 0.8   # More creative
```

### Create Variants
```bash
# Copy base config
cp config/experiments/countdown_deepseek_rl_qwen2_5_vllm.yaml \
   config/experiments/countdown_low_temp.yaml

# Edit the copy with different parameters
# Then run: python run_experiment.py countdown_low_temp
```

## 🚨 **Troubleshooting**

### Common Issues

**1. vLLM Server Not Responding**
```bash
# Check server status
curl http://localhost:8000/health
curl http://localhost:8001/health

# Restart SLURM job if needed
scancel <job_id>
sbatch your_vllm_script.sh
```

**2. Out of Memory Errors**
```bash
# Reduce concurrency
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm --parallel --concurrency 2

# Or run without parallel processing
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm
```

**3. Experiment Hangs**
- Check vLLM server logs: `tail -f qwen2.log deepseek.log`
- Monitor GPU usage: `nvidia-smi`
- Use `--verbose` to see detailed progress

**4. Port Conflicts**
If dashboard port 8080 is busy:
```yaml
# Edit config file
dashboard_port: 8090  # Use different port
```

## 📋 **Step 7: Batch Experiments**

### Run Multiple Experiments
```bash
# Run different configurations
python run_experiment.py countdown_deepseek_rl_qwen2_5_vllm
python run_experiment.py countdown_low_temp
python run_experiment.py countdown_high_iterations

# Use batch script
python -m scripts.batch_run_experiments
```

### Compare Results
```python
import json
import pandas as pd

# Load multiple results
results1 = json.load(open('./results/countdown_deepseek_rl_qwen2_5_vllm/results.json'))
results2 = json.load(open('./results/countdown_low_temp/results.json'))

# Compare success rates
print(f"Base config: {pd.DataFrame(results1)['correct'].mean():.2%}")
print(f"Low temp: {pd.DataFrame(results2)['correct'].mean():.2%}")
```

## 🎯 **Expected Results**

### Typical Performance
- **Countdown problems**: Expect 60-90% success rate depending on problem complexity
- **Processing speed**: ~10-50 problems per minute with parallel processing
- **Token usage**: ~2000-8000 tokens per problem
- **Iterations**: Most problems solved in 1-2 iterations with good models

### Success Metrics to Track
1. **Overall accuracy**: Percentage of problems solved correctly
2. **Iteration efficiency**: How often problems are solved in iteration 1 vs later
3. **Token efficiency**: Average tokens per solved problem
4. **Error patterns**: What types of problems are most challenging

This comprehensive setup allows you to thoroughly evaluate how well your DeepSeek + Qwen2.5 combination performs on Countdown puzzles with iterative reasoning and summarization!