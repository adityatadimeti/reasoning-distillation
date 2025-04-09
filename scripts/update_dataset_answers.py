#!/usr/bin/env python3
"""
Script to update the MC GPQA dataset to use single capital letters (A, B, C, D)
instead of letters with parentheses (A).
"""

import pandas as pd
import re

# Path to the multiple-choice dataset
input_file = 'data/gpqa_diamond_mc.csv'
output_file = 'data/gpqa_diamond_mc.csv'

print(f"Loading dataset from {input_file}")
df = pd.read_csv(input_file)
print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Function to extract letter from format (X)
def extract_letter(answer):
    if not answer:
        return answer
    
    # Extract letter from (A) format
    match = re.search(r'\(([A-Da-d])\)', answer)
    if match:
        return match.group(1).upper()
    
    # If already in correct format (just A, B, C, D)
    if re.fullmatch(r'[A-Da-d]', answer):
        return answer.upper()
    
    # Otherwise, leave as is
    return answer

# Back up the original file
backup_file = input_file + '.bak'
df.to_csv(backup_file, index=False)
print(f"Backed up original dataset to {backup_file}")

# Update the answers
original_answers = df['answer'].copy()
df['answer'] = df['answer'].apply(extract_letter)

# Count changes
changes = sum(original_answers != df['answer'])
print(f"Changed {changes} answers from (X) format to X format")

# Preview the changes
print("\nSample of changes:")
changed_idx = (original_answers != df['answer'])
if changed_idx.any():
    sample = pd.DataFrame({
        'original': original_answers[changed_idx],
        'new': df['answer'][changed_idx]
    }).head(10)
    print(sample)
else:
    print("No changes were necessary")

# Save the updated dataset
df.to_csv(output_file, index=False)
print(f"\nSaved updated dataset to {output_file}")
print("Run the experiment again with the updated dataset format") 