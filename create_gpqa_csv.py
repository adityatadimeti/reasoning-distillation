import pandas as pd
from datasets import load_dataset
import os
import random
from typing import List

# Ensure you are logged in to Hugging Face: `huggingface-cli login`

def format_question_with_choices(question_text: str, choices: List[str]) -> str:
    """Formats the question text with lettered choices."""
    # Strip each choice here during formatting
    formatted_choices = "\n".join([f"({chr(65 + i)}) {choice.strip()}" for i, choice in enumerate(choices)])
    # Also strip the main question text just in case
    return f"{question_text.strip()}\n\nChoices:\n{formatted_choices}"

def create_experiment_csv():
    """Loads the GPQA diamond dataset, formats it, and saves it to CSV."""
    output_path = "data/gpqa_diamond_mc.csv"
    
    try:
        print("Attempting to load the dataset...")
        # Load only the 'train' split
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        print("Dataset loaded successfully.")
        
        processed_data = []
        
        print("Processing dataset...")
        for i, example in enumerate(ds):
            # Extract necessary fields
            question_text = example.get("Question")
            correct_answer_text = example.get("Correct Answer")
            incorrect_answers = [
                example.get("Incorrect Answer 1"),
                example.get("Incorrect Answer 2"),
                example.get("Incorrect Answer 3")
            ]

            # Validate that all parts are present
            if not all([question_text, correct_answer_text] + incorrect_answers):
                 print(f"Warning: Skipping example {i} due to missing data.")
                 continue

            # Combine choices
            all_choices = incorrect_answers + [correct_answer_text]
            
            # --- Start Strip --- 
            # Strip whitespace from each choice *before* shuffling and finding index
            # Handle potential None values just in case
            all_choices_stripped = [str(choice).strip() if choice is not None else "" for choice in all_choices]
            correct_answer_text_stripped = correct_answer_text.strip()
            question_text_stripped = question_text.strip()
            # --- End Strip --- 
            
            # Shuffle the stripped choices
            random.shuffle(all_choices_stripped)
            
            # Find the index and letter of the correct answer in the *stripped* list
            try:
                correct_index = all_choices_stripped.index(correct_answer_text_stripped)
            except ValueError:
                # This might happen if stripping somehow made the correct answer unrecognizable
                print(f"Error: Stripped correct answer '{correct_answer_text_stripped}' not found in stripped choices for original index {i}.")
                print(f"Original correct answer: {repr(correct_answer_text)}")
                print(f"Stripped choices: {all_choices_stripped}")
                continue # Skip this problematic entry

            correct_letter = chr(65 + correct_index) # A=0, B=1, C=2, D=3
            
            # Format the question using the *stripped* question text and *stripped* choices
            # The formatting function itself also applies strip for double safety
            formatted_question = format_question_with_choices(question_text_stripped, all_choices_stripped)
            
            # Create the ID
            # Using a simple index-based ID for now. Adjust if the dataset has a unique ID field.
            problem_id = f"gpqa_diamond_train_{i}" 
            
            # Add to processed list (id, question, solution, answer)
            processed_data.append({
                "id": problem_id,
                "question": formatted_question,
                "solution": correct_answer_text_stripped, # Use the stripped solution
                "answer": correct_letter
            })

        if not processed_data:
             print("No data processed. Exiting.")
             return

        # Create DataFrame and save
        df = pd.DataFrame(processed_data)
        
        print(f"\nSaving formatted data to {output_path}...")
        df.to_csv(output_path, index=False)
        print("Formatted data saved successfully.")
        print(f"\nFirst few rows of the generated CSV ({output_path}):")
        print(df.head())

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you are logged in to Hugging Face (`huggingface-cli login`),")
        print("have the 'datasets' and 'pandas' libraries installed,")
        print("and have internet connectivity.")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    create_experiment_csv() # Changed function call
