"""
Dataset loading and management for reasoning enhancement project.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.utils.config import Config, PROJECT_ROOT

class Dataset:
    """
    Class to load and manage datasets for reasoning evaluation.
    """
    def __init__(self, config: Config):
        """
        Initialize the dataset manager.
        
        Args:
            config: Configuration object with dataset settings
        """
        self.config = config
        self.dataset_config = config.get("dataset", {})
        
        # Dataset metadata
        self.name = self.dataset_config.get("name", "unknown")
        self.description = self.dataset_config.get("description", "")
        
        # File paths
        self.base_path = Path(self.dataset_config.get("base_path", "data"))
        if not self.base_path.is_absolute():
            self.base_path = PROJECT_ROOT / self.base_path
            
        self.raw_file = self.dataset_config.get("raw_file", "")
        self.raw_path = self.base_path / self.raw_file
        
        # For archive data
        self.archive_path = PROJECT_ROOT / "archive" / "data"
        
        # Check for data in archive if not in base path
        if not self.raw_path.exists() and (self.archive_path / self.raw_file).exists():
            self.raw_path = self.archive_path / self.raw_file
        
        # Processed data paths
        self.processed_path = self.base_path / "processed"
        self.processed_file = f"{self.name}_processed.parquet"
        
        # Column mappings
        self.columns = self.dataset_config.get("columns", {})
        
        # Processing options
        self.use_cached = self.dataset_config.get("use_cached", True)
        self.train_test_split = self.dataset_config.get("train_test_split", 0.8)
        self.random_seed = self.dataset_config.get("random_seed", 42)
        
        # Initialize data storage
        self.data_raw = None
        self.data_processed = None
        self.train_data = None
        self.test_data = None
    
    def load_raw(self) -> pd.DataFrame:
        """
        Load the raw dataset from disk.
        
        Returns:
            Raw dataset as a pandas DataFrame
            
        Raises:
            FileNotFoundError: If the raw data file doesn't exist
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_path}")
        
        # Load based on file extension
        if self.raw_path.suffix.lower() == '.csv':
            self.data_raw = pd.read_csv(self.raw_path)
        elif self.raw_path.suffix.lower() == '.parquet':
            self.data_raw = pd.read_parquet(self.raw_path)
        else:
            raise ValueError(f"Unsupported file format: {self.raw_path.suffix}")
        
        return self.data_raw
    
    def load_processed(self, force_reprocess: bool = False) -> pd.DataFrame:
        """
        Load the processed dataset, processing raw data if necessary.
        
        Args:
            force_reprocess: If True, reprocess raw data even if cached data exists
            
        Returns:
            Processed dataset as a pandas DataFrame
        """
        processed_file_path = self.processed_path / self.processed_file
        
        # Load cached processed data if available and requested
        if not force_reprocess and self.use_cached and processed_file_path.exists():
            self.data_processed = pd.read_parquet(processed_file_path)
            print(f"Loaded processed data from {processed_file_path}")
            return self.data_processed
        
        # Otherwise, load raw data and process it
        if self.data_raw is None:
            self.load_raw()
        
        # Process the data
        from src.data.preprocessing import preprocess_dataset
        self.data_processed = preprocess_dataset(self.data_raw, self.columns)
        
        # Save processed data
        if not self.processed_path.exists():
            self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.data_processed.to_parquet(processed_file_path)
        print(f"Saved processed data to {processed_file_path}")
        
        return self.data_processed
    
    def split_train_test(self, test_size: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the processed dataset into training and testing sets.
        
        Args:
            test_size: Fraction of data to use for testing (overrides config)
            
        Returns:
            Tuple of (train_data, test_data) as pandas DataFrames
        """
        if self.data_processed is None:
            self.load_processed()
        
        from sklearn.model_selection import train_test_split
        
        test_fraction = test_size if test_size is not None else (1.0 - self.train_test_split)
        
        self.train_data, self.test_data = train_test_split(
            self.data_processed,
            test_size=test_fraction,
            random_state=self.random_seed
        )
        
        return self.train_data, self.test_data
    
    def get_problems(self, split: str = "all") -> List[Dict]:
        """
        Get problems formatted as dictionaries for easy use in models.
        
        Args:
            split: Which data split to use ('all', 'train', or 'test')
            
        Returns:
            List of problem dictionaries
        """
        if split == "train" and self.train_data is None:
            self.split_train_test()
        
        if split == "test" and self.test_data is None:
            self.split_train_test()
        
        data = {
            "all": self.data_processed,
            "train": self.train_data,
            "test": self.test_data
        }.get(split)
        
        if data is None:
            raise ValueError(f"Invalid split: {split}. Must be 'all', 'train', or 'test'.")
        
        # Convert to list of dictionaries
        problems = []
        for _, row in data.iterrows():
            problem = {
                "id": row["id"],
                "question": row["question"],
                "reasoning_trace": row.get("reasoning_trace", ""),
                "ground_truth": row.get("ground_truth", ""),
                "extracted_answer": row.get("extracted_answer", "")
            }
            problems.append(problem)
        
        return problems
    
    def get_problem_by_id(self, problem_id: str) -> Dict:
        """
        Get a specific problem by ID.
        
        Args:
            problem_id: ID of the problem to retrieve
            
        Returns:
            Problem as a dictionary
            
        Raises:
            ValueError: If problem_id is not found
        """
        if self.data_processed is None:
            self.load_processed()
        
        matching_rows = self.data_processed[self.data_processed["id"] == problem_id]
        
        if len(matching_rows) == 0:
            raise ValueError(f"Problem ID not found: {problem_id}")
        
        row = matching_rows.iloc[0]
        problem = {
            "id": row["id"],
            "question": row["question"],
            "reasoning_trace": row.get("reasoning_trace", ""),
            "ground_truth": row.get("ground_truth", ""),
            "extracted_answer": row.get("extracted_answer", "")
        }
        
        return problem