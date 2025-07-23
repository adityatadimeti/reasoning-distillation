#!/bin/bash
#SBATCH --job-name=harp_deepseek_qwen_14b_pass
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --partition=cocoflops-interactive
#SBATCH --account=cocoflops
#SBATCH --nodelist=cocoflops1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
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
python3 run_experiment.py harp_deepseek_qwen_14b_pass \
  --parallel --concurrency 8 \
  --question_ids H6-133,H6-087,H6-035,H6-051,H6-131,H6-059,H6-115,H6-187,H6-097,H6-052,H6-179,H6-053,H6-169,H6-082,H6-180,H6-126,H6-105,H6-031,H6-140,H6-093,H6-070,H6-124,H6-165,H6-190,H6-116,H6-145,H6-182,H6-069,H6-128,H6-102,H6-125,H6-132,H6-152,H6-130,H6-013,H6-101,H6-104,H6-197,H6-177,H6-181,H6-139,H6-100,H6-192,H6-183,H6-161,H6-164,H6-110,H6-099,H6-174,H6-078,H6-184,H6-170,H6-122,H6-148,H6-142,H6-072,H6-095,H6-154,H6-171,H6-112,H6-194,H6-196,H6-147,H6-155,H6-168,H6-119,H6-136,H6-167,H6-150,H6-103,H6-127,H6-191,H6-143,H6-135,H6-061,H6-172,H6-158,H6-163,H6-173,H6-106,H6-118,H6-151,H6-137,H6-153,H6-176,H6-185,H6-056,H6-175,H6-081,H6-144,H6-166,H6-023,H6-117,H6-195,H6-138,H6-098,H6-193,H6-141,H6-113,H6-123,H6-044,H6-157,H6-186,H6-079,H6-108,H6-046,H6-109,H6-012,H6-178,H6-120,H6-121,H6-146,H6-073,H6-038,H6-015,H6-114,H6-189,H6-006,H6-156,H6-134,H6-111,H6-054,H6-075,H6-089,H6-162,H6-160,H6-107,H6-094,H6-129,H6-188,H6-159,H6-149








