#!/bin/bash
#SBATCH --job-name=harp_deepseek_qwen_14b_pass_base_summ
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --partition=cocoflops-interactive
#SBATCH --account=cocoflops
#SBATCH --nodelist=cocoflops1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nice=0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jshen3@stanford.edu

# Move to project directory
cd /afs/cs.stanford.edu/u/jshen3/research_projects/reasoning-distillation

# Load conda environment
source ~/.bashrc
conda activate distill

# Make sure logs directory exists
mkdir -p logs

# Run your experiment
python3 run_experiment.py harp_deepseek_qwen_14b_summ_base_sum_4iter_l6 \
  --parallel --concurrency 8 \
  --question_ids H6-193,H6-140,H6-165,H6-146,H6-167,H6-129,H6-168,H6-152,H6-163,H6-141















