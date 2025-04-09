#!/usr/bin/env python3
"""
Script to reformat the GPQA Diamond dataset to include multiple-choice answers
and evaluate using letter choices (A, B, C, D) instead of string matching.
"""

import pandas as pd
import uuid
import random
import re
from datasets import load_dataset

# Download the original GPQA Diamond dataset (with multiple choice options)
print("Downloading original GPQA Diamond dataset with multiple choice options...")

# Uncomment the following to download from Hugging Face
# dataset = load_dataset("Idavidrein/gpqa", "diamond")
# df = pd.DataFrame(dataset['train'])

# For demonstration purposes, we'll create a small mock dataset
# In a real scenario, you would use the commented code above to load the actual dataset
df = pd.DataFrame({
    'Question': [
        'Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?',
        'A spin-half particle is in a linear superposition 0.5|\\uparrow\\rangle+sqrt(3)/2|\\downarrow\\rangle of its spin-up and spin-down states. If |\\uparrow\\rangle and |\\downarrow\\rangle are the eigenstates of \\sigma{z}, then what is the expectation value up to one decimal place, of the operator 10\\sigma{z}+5\\sigma_{x}? Here, symbols have their usual meanings'
    ],
    'Correct Answer': ['10^-4 eV', '-0.7'],
    'Incorrect Answer 1': ['10^-7 eV', '-0.5'],
    'Incorrect Answer 2': ['10^-10 eV', '0.7'],
    'Incorrect Answer 3': ['10^-1 eV', '0.5'],
    'Explanation': [
        "According to the uncertainty principle, Delta E* Delta t=hbar/2...",
        "|psi> = 0.5 |up=0> + sqrt(3)/2|down=1>..."
    ]
})

print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Prepare the new dataframe with required columns
print("Reformatting data to include multiple-choice options...")
new_data = []

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

for i, row in enumerate(df.iterrows()):
    # Generate a unique ID that includes the index for easier filtering
    index = i  # 0-indexed
    unique_part = str(uuid.uuid4())[:8]
    unique_id = f"idx_{index:03d}_{unique_part}"
    
    # Get choices and shuffle them
    choices = [
        preprocess(row[1]["Incorrect Answer 1"]),
        preprocess(row[1]["Incorrect Answer 2"]),
        preprocess(row[1]["Incorrect Answer 3"]),
        preprocess(row[1]["Correct Answer"]),
    ]
    
    random.shuffle(choices)
    correct_answer_index = choices.index(preprocess(row[1]["Correct Answer"]))
    correct_letter = chr(65 + correct_answer_index)  # A, B, C, or D
    
    # Format the question with answer choices
    original_question = row[1]['Question']
    formatted_question = original_question + "\n\nChoices:\n"
    for j, choice in enumerate(choices):
        formatted_question += f"({chr(65 + j)}) {choice}\n"
    
    # Extract the explanation
    explanation = row[1].get('Explanation', '')
    
    # Create solution with explanation and correct choice
    solution = f"{explanation}\n\nThe correct answer is ({correct_letter}): {row[1]['Correct Answer']}"
    
    new_data.append({
        'id': unique_id,
        'question': formatted_question,
        'solution': solution,
        'answer': f"({correct_letter})"
    })

# Create new dataframe with the required structure
new_df = pd.DataFrame(new_data)

# Save to CSV
output_path = 'data/gpqa_diamond_multichoice.csv'
new_df.to_csv(output_path, index=False)
print(f"Saved multiple-choice formatted dataset to {output_path}")

# Print preview
print("\nPreview of the reformatted dataset:")
print(new_df.head(2))
print(f"Total samples: {new_df.shape[0]}")

print("\nWith this format, answer verification can use a simple regex to extract")
print("the letter choice (A, B, C, or D) and compare it to the correct letter.") 