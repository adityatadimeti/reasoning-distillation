#!/bin/bash

echo "Running Seed 2 experiments..."

# Post think
python run_experiment.py config/experiments/aime_deepseek_qwen_14b_baseline_post_think_sum_4iter_seed_2.yaml --parallel --concurrency 12 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_2/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_2_20250513_002335/results.json

# Base summary
python run_experiment.py config/experiments/aime_deepseek_qwen_14b_summ_base_sum_4iter_seed_2.yaml --parallel --concurrency 12 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_2/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_2_20250513_002335/results.json

# First k
python run_experiment.py config/experiments/aime_deepseek_qwen_14b_baseline_firstk_sum_4iter_seed_2.yaml --parallel --concurrency 12 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_2/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_2_20250513_002335/results.json

echo "Seed 2 experiments completed!" 