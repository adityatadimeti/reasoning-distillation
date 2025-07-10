#!/usr/bin/env python3
"""
Convert Countdown dataset from Huggingface parquet format to CSV format.

This script loads the Countdown-Tasks-3to4 dataset from Huggingface and converts it
to the CSV format expected by the reasoning distillation codebase.

Usage:
    python scripts/convert_countdown_to_csv.py
"""

import pandas as pd
import os
from pathlib import Path

def format_question(nums, target):
    """
    Format the Countdown problem into a natural language question.
    
    Args:
        nums: List of available numbers
        target: Target number to reach
        
    Returns:
        Formatted question string
    """
    nums_str = ", ".join(map(str, nums))
    return f"Using the numbers {nums_str}, reach the target number {target}. You may use addition (+), subtraction (-), multiplication (*), and division (/). Each number can be used at most once."

def main():
    """Main conversion function."""
    print("Loading Countdown dataset from Huggingface...")
    
    # Load the dataset
    df = pd.read_parquet("hf://datasets/Jiayi-Pan/Countdown-Tasks-3to4/data/train-00000-of-00001.parquet")
    
    print(f"Loaded {len(df)} problems")
    
    # Create output directory if it doesn't exist
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Convert to required format
    converted_data = []
    
    for idx, row in df.iterrows():
        # Extract nums and target
        nums = row['nums']
        target = row['target']
        
        # Create ID
        problem_id = f"countdown_{idx+1}"
        
        # Format question
        question = format_question(nums, target)
        
        # For solution, we'll leave it empty as it needs to be generated
        # The answer is the target number itself
        solution = ""
        answer = str(target)
        
        converted_data.append({
            'id': problem_id,
            'question': question,
            'solution': solution,
            'answer': answer
        })
    
    # Create DataFrame
    converted_df = pd.DataFrame(converted_data)
    
    # Save to CSV
    output_path = output_dir / "countdown.csv"
    converted_df.to_csv(output_path, index=False)
    
    print(f"Converted {len(converted_df)} problems and saved to {output_path}")
    
    # Show sample
    print("\nSample problems:")
    print(converted_df.head())
    
    # Save metadata about the conversion
    metadata = {
        'source': 'Jiayi-Pan/Countdown-Tasks-3to4',
        'total_problems': len(converted_df),
        'description': 'Countdown arithmetic puzzle problems where you combine numbers to reach a target',
        'rules': [
            'Use addition (+), subtraction (-), multiplication (*), and division (/)',
            'Each number can be used at most once',
            'Must reach exactly the target number'
        ]
    }
    
    import json
    metadata_path = output_dir / "countdown_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_path}")

if __name__ == "__main__":
    main()