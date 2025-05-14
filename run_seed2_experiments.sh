#!/bin/bash

echo "Running Seed 2 experiments..."

# # Base summary
# python run_experiment.py aime_deepseek_qwen_14b_summ_base_sum_4iter_seed_2 --parallel --concurrency 8 \
#   --load_initial_reasoning=results/aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_2/aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_2_20250513_121629/results.json

# Post think
python run_experiment.py aime_deepseek_qwen_14b_baseline_post_think_sum_4iter_seed_2 --parallel --concurrency 8 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_2/aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_2_20250513_121629/results.json

# First k
python run_experiment.py aime_deepseek_qwen_14b_baseline_firstk_sum_4iter_seed_2 --parallel --concurrency 8 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_2/aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_2_20250513_121629/results.json

echo "Seed 2 experiments completed!"