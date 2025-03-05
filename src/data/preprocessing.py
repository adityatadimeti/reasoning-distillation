"""
Data preprocessing functions for reasoning enhancement project.
"""
import pandas as pd
import re
from typing import Dict, List, Optional, Union

def preprocess_dataset(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Preprocess the raw dataset with standard cleaning steps.
    
    Args:
        df: Raw dataframe
        column_mapping: Dictionary mapping standardized column names to actual column names
        
    Returns:
        Processed dataframe with standardized columns
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Standardize column names
    # Map from configured column names to standardized internal names
    standard_columns = {
        "id": column_mapping.get("id", "ID"),
        "question": column_mapping.get("problem", "Problem"),
        "solution": column_mapping.get("solution", "Solution"),
        "ground_truth": column_mapping.get("answer", "Answer"),
        "reasoning_trace": column_mapping.get("reasoning_trace", "model_completion"),
        "extracted_answer": column_mapping.get("extracted_answer", "extracted_answer")
    }
    
    # Rename columns
    column_rename = {}
    for standard_name, original_name in standard_columns.items():
        if original_name in processed_df.columns:
            column_rename[original_name] = standard_name
    
    processed_df = processed_df.rename(columns=column_rename)
    
    # Ensure all required columns exist
    required_columns = ["id", "question"]
    missing_columns = [col for col in required_columns if col not in processed_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean and standardize question text
    if "question" in processed_df.columns:
        processed_df["question"] = processed_df["question"].apply(clean_text)
    
    # Extract ground truth if needed
    if "ground_truth" in processed_df.columns:
        processed_df["ground_truth"] = processed_df["ground_truth"].apply(extract_ground_truth)
    
    # Normalize answers for consistency
    if "extracted_answer" in processed_df.columns:
        processed_df["normalized_extracted"] = processed_df["extracted_answer"].apply(normalize_answer)
    
    if "ground_truth" in processed_df.columns:
        processed_df["normalized_ground_truth"] = processed_df["ground_truth"].apply(normalize_answer)
    
    return processed_df

def clean_text(text: str) -> str:
    """
    Clean and standardize text fields.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Fix common LaTeX issues
    text = text.replace('\\\\', '\\')  # Double backslashes
    
    return text

def extract_ground_truth(answer_text: str) -> str:
    """
    Extract the ground truth answer.
    
    Args:
        answer_text: Text containing the answer
        
    Returns:
        Extracted ground truth
    """
    if not isinstance(answer_text, str):
        return str(answer_text) if answer_text is not None else ""
    
    # Check for the pattern '#### X'
    pattern = r'####\s*(.*?)$'
    match = re.search(pattern, answer_text)
    
    if match:
        return match.group(1).strip()
    
    return answer_text.strip()

def normalize_answer(answer: Union[str, int, float]) -> str:
    """
    Normalize answers for consistent comparison.
    
    Args:
        answer: Answer to normalize
        
    Returns:
        Normalized answer
    """
    if not isinstance(answer, str):
        return str(answer) if answer is not None else ""
    
    # Remove whitespace and convert to lowercase
    answer = answer.strip().lower()
    
    # Remove commas from numbers
    answer = re.sub(r'(\d),(\d)', r'\1\2', answer)
    
    # Remove dollar signs, percentage signs, etc.
    answer = re.sub(r'[$%]', '', answer)
    
    # Try to extract just the numerical part if it's a complex string
    numeric_match = re.search(r'[-+]?\d*\.?\d+', answer)
    if numeric_match:
        number_str = numeric_match.group(0)
        number = float(number_str)
        
        # Convert to integer if it's a whole number
        if number.is_integer():
            return str(int(number))
        return number_str
    
    return answer