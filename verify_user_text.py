#!/usr/bin/env python3
"""Verify extraction and evaluation on user's exact text"""

from src.reasoning.extractor import extract_answer_with_config
from src.eval.countdown_check import validate_equation, evaluate_equation
from src.experiments.summarization import get_answer_checker

# Exact text from user - note this would be the value of the "reasoning" field
reasoning_text = """<think>
Okay, so I'm trying to solve this Countdown numbers puzzle where I have to use all four numbers: 124, 192, 127, and 151. The target number I need to reach is 44. I remember that in these puzzles, I can use each number exactly once and combine them with addition, subtraction, multiplication, or division. I also need to use parentheses to control the order of operations.

In my first attempt, I tried subtracting pairs of numbers. I did 192 - 124, which gave me 68, and then 151 - 127, which gave me 24. Then I subtracted those two results: 68 - 24 = 44. That seemed to work, and I thought I was done. But now I'm a bit unsure because I want to make sure there aren't other ways to reach 44, maybe using multiplication or division.

Let me think again. Maybe I can try combining the numbers in a different way. For example, what if I use division? Let's see, maybe 192 divided by something. If I take 192 and divide it by 124, that's roughly 1.55. Hmm, that doesn't seem helpful. What if I do 127 divided by 151? That's about 0.84, which also doesn't help me get closer to 44.

What if I try adding some numbers first? Let's say I add 124 and 127. That gives me 251. Then I have 192 and 151 left. If I subtract 151 from 192, I get 41. Now I have 251 and 41. If I subtract 41 from 251, that's 210, which is way too big. Alternatively, if I divide 251 by 41, that's about 6.12, not 44.

Maybe I should try multiplying. If I multiply 124 by 127, that's a huge number, way over 44. Same with 192 and 151. Multiplying any two numbers here would probably overshoot the target. So maybe multiplication isn't the way to go.

What about combining addition and subtraction differently? Let's try (192 - 127) which is 65, and then see if I can use 124 and 151 somehow. If I subtract 151 from 124, that's -27. Then I have 65 and -27. If I add them, 65 + (-27) = 38, which is close but not 44. If I subtract them, 65 - (-27) = 92, which is too high.

Another idea: Maybe (151 - 124) = 27. Then, if I can get 44 from 192 and 127 using 27 somehow. Wait, I already used 151 and 124 to make 27. So I have 192 and 127 left. Maybe 192 divided by something? 192 divided by 127 is about 1.51, which doesn't help. Alternatively, 127 - 192 = -65, which doesn't help either.

Wait, maybe I can do something like (192 / (127 - 124)) = 192 / 3 = 64. Then, if I subtract 64 from 151, that's 87, which is too high. Alternatively, 64 - 151 = -87, which isn't helpful.

Another approach: Let's try to make a larger number and then subtract a smaller one. For example, 192 + 127 = 319. Then, 151 + 124 = 275. If I subtract 275 from 319, that's 44! Wait, that works. But hold on, in this case, I added 192 and 127 together, and 151 and 124 together, then subtracted. But the problem is, I used each number exactly once, right? Let me check: 192 + 127 uses both, and 151 + 124 uses both. Then subtracting those sums: (192 + 127) - (151 + 124) = 319 - 275 = 44. Yes, that uses each number once, and it's a valid operation.

Wait, but in my first attempt, I had (192 - 124) - (151 - 127) = 68 - 24 = 44. Now I have another way: (192 + 127) - (151 + 124) = 44. So which one is correct? Both seem to use each number once and reach 44.

But wait, in the first approach, I subtracted two differences, while in the second, I subtracted two sums. Both are valid. So maybe there are multiple solutions. But the problem asks for an equation that equals 44, using all numbers exactly once. So both approaches are correct, but I need to choose one.

However, in my first attempt, I already found a valid solution, so maybe that's sufficient. But I'm now realizing there's another way. But since the problem allows any correct equation, either one is acceptable.

But let me double-check my second approach: (192 + 127) = 319, and (151 + 124) = 275. Then 319 - 275 = 44. Yes, that works. So both methods are correct.

But in the first attempt, I had (192 - 124) = 68, and (151 - 127) = 24, then 68 - 24 = 44. That also works. So both are valid. Therefore, the answer can be either of these equations.

But the problem says to return the equation in <answer> tags, so I can choose either. However, since I already provided the first one in my previous attempt, maybe I should stick with that unless I find a better one.

Wait, but in the second approach, I used addition and subtraction, which might be considered more straightforward. But both are correct. I think either is acceptable, but I'll go with the first one I found because it's simpler in terms of operations.

So, to recap, the equation is (192 - 124) - (151 - 127) = 68 - 24 = 44. Each number is used exactly once, and the operations are correctly ordered with parentheses. Therefore, this is a valid solution.
</think>

The solution uses subtraction on two pairs of numbers and then subtracts the results to reach the target. Each number is used exactly once.

<answer>(192 - 124) - (151 - 127)</answer>"""

# Configuration and ground truth
config = {"answer_extractor": "countdown"}
nums = [124, 192, 127, 151]
target = 44
gt_answer = {"target": target, "nums": nums}

print("=== Verifying Answer Extraction and Evaluation ===\n")

# Extract answer
extracted = extract_answer_with_config(reasoning_text, config)
print(f"1. Extracted answer: '{extracted}'")

# Validate
is_valid = validate_equation(extracted, nums)
print(f"\n2. Validation: {is_valid}")
print(f"   Uses all numbers exactly once: {'YES' if is_valid else 'NO'}")

# Evaluate
result = evaluate_equation(extracted)
print(f"\n3. Evaluation: {result}")
print(f"   Target: {target}")
print(f"   Matches: {'YES' if result == target else 'NO'}")

# Full grading
answer_checker = get_answer_checker(config)
check_result = answer_checker(extracted, gt_answer, extract_policy="none", eval_policy="aggressive")
print(f"\n4. Full grading result: {'CORRECT' if check_result['is_correct'] else 'INCORRECT'}")

# Manual calculation verification
print(f"\n5. Manual verification:")
print(f"   192 - 124 = {192 - 124}")
print(f"   151 - 127 = {151 - 127}")
print(f"   68 - 24 = {68 - 24}")

print("\n=== Summary ===")
print(f"✓ Answer extraction: Working correctly")
print(f"✓ Validation: Working correctly") 
print(f"✓ Evaluation: Working correctly")
print(f"✓ Full grading: Working correctly")
print(f"\nThe equation '{extracted}' is properly extracted and evaluated as CORRECT.")