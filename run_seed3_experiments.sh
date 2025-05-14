#!/bin/bash

echo "Running Seed 3 experiments..."

# # First k
# python run_experiment.py aime_deepseek_qwen_14b_baseline_firstk_sum_4iter_seed_3 --parallel --concurrency 12 \
#   --load_initial_reasoning=results/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3_20250513_110758/results.json

# Last k
python run_experiment.py aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_3 --parallel --concurrency 8 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3_20250513_110758/results.json

# Post think
python run_experiment.py aime_deepseek_qwen_14b_baseline_post_think_sum_4iter_seed_3 --parallel --concurrency 8 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3_20250513_110758/results.json

# Base summary
python run_experiment.py aime_deepseek_qwen_14b_summ_base_sum_4iter_seed_3 --parallel --concurrency 8 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3_20250513_110758/results.json

echo "Seed 3 experiments completed!"