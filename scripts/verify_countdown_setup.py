#!/usr/bin/env python3
"""Verify countdown experiment setup is correct"""

import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.summarization import get_answer_checker
from src.eval.countdown_check import check_one_countdown_answer, validate_equation, evaluate_equation

print("=== Verifying Countdown Setup ===\n")

# Test 1: Check if answer checker selection works
print("1. Testing answer checker selection:")
config_countdown = {"answer_extractor": "countdown"}
checker = get_answer_checker(config_countdown)
print(f"   ✓ Countdown checker created: {checker}")

# Test 2: Test direct evaluation
print("\n2. Testing direct countdown evaluation:")
equation = "(192 - 124) - (151 - 127)"
nums = [124, 192, 127, 151]
target = 44

print(f"   Equation: {equation}")
print(f"   Numbers: {nums}")
print(f"   Target: {target}")

# Validate
is_valid = validate_equation(equation, nums)
print(f"   Validation: {is_valid}")

# Evaluate
result = evaluate_equation(equation)
print(f"   Result: {result}")
print(f"   Correct: {result == target}")

# Test 3: Check with answer tags
print("\n3. Testing with answer tags:")
answer_with_tags = f"<answer>{equation}</answer>"
check_result = check_one_countdown_answer(answer_with_tags, nums, target)
print(f"   Is correct: {check_result['is_correct']}")
if 'error' in check_result:
    print(f"   Error: {check_result['error']}")

# Test 4: Test the answer checker function
print("\n4. Testing answer checker with ground truth dict:")
gt_answer = {"target": target, "nums": nums}
checker_result = checker(equation, gt_answer, extract_policy="none", eval_policy="aggressive")
print(f"   Is correct: {checker_result['is_correct']}")

# Load and check experiment config
print("\n5. Checking experiment config:")
config_path = Path("config/experiments/countdown_deepseek_rl_qwen2_5_vllm.yaml")
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"   answer_extractor: {config.get('answer_extractor', 'NOT SET')}")
    print(f"   max_tokens: {config.get('max_tokens')}")
    print(f"   summary_max_tokens: {config.get('summary_max_tokens')}")
else:
    print(f"   ❌ Config file not found: {config_path}")

print("\n=== Setup Verification Complete ===")
print("\nIf all checks pass but the experiment still grades incorrectly,")
print("make sure to sync these files to your remote machine:")
print("  - src/experiments/summarization.py")
print("  - src/eval/countdown_check.py")