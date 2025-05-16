#!/bin/bash

# # seed 1
# python run_experiment.py aime_deepseek_qwen_14b_baseline_answer_only_sum_4iter_seed_1 --parallel --concurrency 8 \
#   --load_initial_reasoning=results/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_1/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_1_20250512_202741/merged_results_20250513_220116.json

# seed 2
python run_experiment.py aime_deepseek_qwen_14b_baseline_answer_only_sum_4iter_seed_2 --parallel --concurrency 8 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_2/aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_2_20250513_121629/results.json \
  --question_ids=14,16,5,28,9,19,18,29,2,20,24,15,11,22,17,27,21,23,10,26,13,12,25

# seed 3
python run_experiment.py aime_deepseek_qwen_14b_baseline_answer_only_sum_4iter_seed_3 --parallel --concurrency 8 \
  --load_initial_reasoning=results/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3/aime_deepseek_qwen_14b_summ_base_sum_4iter_backtracking_seed_3_20250513_110758/results.json