import pandas as pd
import uuid
import random
import re

# Load the existing GPQA Diamond dataset
print("Loading existing GPQA Diamond dataset...")
df = pd.read_csv('data/gpqa_diamond.csv')
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Prepare the new dataframe with multiple-choice format
print("Reformatting data to multiple-choice format...")
new_data = []

# Helper function to generate plausible incorrect answers
def generate_incorrect_answers(correct_answer, num_needed=3):
    """Generate plausible but incorrect answers for multiple choice questions."""
    # For numeric answers
    if re.search(r'\d', correct_answer):
        # Is it scientific notation?
        if '^' in correct_answer or 'e' in correct_answer.lower():
            # Generate variations by changing the exponent
            match = re.search(r'(\d+(?:\.\d+)?)[eE^]([+-]?\d+)', correct_answer)
            if match:
                base = float(match.group(1))
                exp = int(match.group(2))
                # Create variations by adjusting the exponent
                variations = [
                    f"{base}^{exp-3}" if '^' in correct_answer else f"{base}e{exp-3}",
                    f"{base}^{exp+3}" if '^' in correct_answer else f"{base}e{exp+3}",
                    f"{base*10}^{exp-1}" if '^' in correct_answer else f"{base*10}e{exp-1}"
                ]
                return variations[:num_needed]
        
        # Is it a simple number?
        try:
            num = float(re.search(r'-?\d+(?:\.\d+)?', correct_answer).group())
            unit = re.search(r'[a-zA-Z]+', correct_answer)
            unit = unit.group() if unit else ""
            
            # Generate variations
            variations = [
                f"{num/10} {unit}".strip(),
                f"{num*10} {unit}".strip(),
                f"{-num} {unit}".strip(),
                f"{num+1} {unit}".strip()
            ]
            return variations[:num_needed]
        except:
            pass
    
    # For text answers
    # Generate some generic incorrect answers
    return [
        "None of the above",
        "All of the above",
        "Cannot be determined from the given information"
    ][:num_needed]

for i, row in enumerate(df.iterrows()):
    # Get the existing ID, question, and correct answer
    unique_id = row[1]['id']
    question = row[1]['question']
    correct_answer = row[1]['answer']
    solution = row[1]['solution']
    
    # Generate incorrect answers
    incorrect_answers = generate_incorrect_answers(correct_answer)
    
    # Ensure we have exactly 3 incorrect answers
    while len(incorrect_answers) < 3:
        incorrect_answers.append(f"Incorrect option {len(incorrect_answers) + 1}")
    
    # Create choices and shuffle them
    choices = incorrect_answers + [correct_answer]
    random.shuffle(choices)
    
    # Find the index of the correct answer
    correct_index = choices.index(correct_answer)
    correct_letter = chr(65 + correct_index)  # A, B, C, or D
    
    # Format the question with choices
    formatted_question = question + "\n\nChoices:\n"
    for j, choice in enumerate(choices):
        formatted_question += f"({chr(65 + j)}) {choice}\n"
    
    # Update the solution to include the letter
    formatted_solution = solution + f"\n\nThe correct answer is ({correct_letter})."
    
    new_data.append({
        'id': unique_id,
        'question': formatted_question,
        'solution': formatted_solution,
        'answer': f"({correct_letter})"  # Store the answer in (X) format
    })

# Create new dataframe
new_df = pd.DataFrame(new_data)

# Save to CSV
output_path = 'data/gpqa_diamond_mc.csv'
new_df.to_csv(output_path, index=False)
print(f"Saved multiple-choice dataset to {output_path}")

# Print preview
print("\nPreview of the first entry:")
print(f"ID: {new_df.iloc[0]['id']}")
print(f"Question: {new_df.iloc[0]['question'][:200]}...")
print(f"Answer: {new_df.iloc[0]['answer']}")
print(f"\nTotal samples: {new_df.shape[0]}") 