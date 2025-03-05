"""
Tests for dataset loading and preprocessing functionality.
"""
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.utils.config import Config, PROJECT_ROOT
from src.data.dataset import Dataset
import src.data.preprocessing as preprocessing

# Sample data for testing
SAMPLE_DATA = {
    "ID": ["2024-I-1", "2024-I-2", "2024-I-3"],
    "Problem": [
        "If x + y = 10, what is x × y?",
        "Solve for x: 2x + 5 = 15",
        "A train travels at 60mph for 3 hours. How far does it go?"
    ],
    "Solution": [
        "Using the identity (x+y)² = x² + 2xy + y², we get...",
        "Subtracting 5 from both sides: 2x = 10...",
        "Distance = speed × time = 60mph × 3h = 180 miles"
    ],
    "Answer": ["25", "5", "180 miles"],
    "model_completion": [
        "I'll solve this step by step. Let x + y = 10, so y = 10 - x.\nThen x × y = x × (10 - x) = 10x - x² = x(10 - x).\nTo maximize this product, we can use calculus or the AM-GM inequality.\nBy AM-GM, the product is maximized when x = y = 5.\nSo the maximum value of x × y is 5 × 5 = 25.\n\nThe answer is \\boxed{25}.",
        "2x + 5 = 15\nSubtracting 5 from both sides:\n2x = 10\nDividing by 2:\nx = 5\n\nThe answer is \\boxed{5}.",
        "Distance = speed × time\nDistance = 60 mph × 3 hours\nDistance = 180 miles\n\nThe answer is \\boxed{180}."
    ],
    "extracted_answer": ["25", "5", "180"]
}

class TestPreprocessing:
    """Tests for preprocessing functions"""
    
    def test_clean_text(self):
        """Test text cleaning function"""
        # Test removing excess whitespace
        assert preprocessing.clean_text("  text  with   spaces  ") == "text with spaces"
        
        # Test handling non-string input
        assert preprocessing.clean_text(123) == "123"
        
        # Test LaTeX backslash fixing
        assert preprocessing.clean_text("$x \\\\times y$") == "$x \\times y$"
    
    def test_extract_ground_truth(self):
        """Test ground truth extraction"""
        # Test exact pattern
        assert preprocessing.extract_ground_truth("Some explanation\n#### 42") == "42"
        
        # Test with spacing variations
        assert preprocessing.extract_ground_truth("Explanation\n####   123  ") == "123"
        
        # Test when pattern not present
        assert preprocessing.extract_ground_truth("Just an answer: 7") == "Just an answer: 7"
        
        # Test with non-string input
        assert preprocessing.extract_ground_truth(42) == "42"
    
    def test_normalize_answer(self):
        """Test answer normalization"""
        # Test string numbers
        assert preprocessing.normalize_answer("42") == "42"
        
        # Test with units and symbols
        assert preprocessing.normalize_answer("$180 miles") == "180"
        
        # Test with commas
        assert preprocessing.normalize_answer("1,234") == "1234"
        
        # Test decimal to integer conversion
        assert preprocessing.normalize_answer("42.0") == "42"
        
        # Test float preservation
        assert preprocessing.normalize_answer("3.14") == "3.14"
        
        # Test with non-string input
        assert preprocessing.normalize_answer(42) == "42"
        assert preprocessing.normalize_answer(3.14) == "3.14"
    
    def test_preprocess_dataset(self):
        """Test dataset preprocessing function"""
        # Create sample DataFrame
        df = pd.DataFrame(SAMPLE_DATA)
        
        # Define column mapping
        column_mapping = {
            "id": "ID",
            "problem": "Problem",
            "solution": "Solution",
            "answer": "Answer",
            "reasoning_trace": "model_completion",
            "extracted_answer": "extracted_answer"
        }
        
        # Preprocess the DataFrame
        processed_df = preprocessing.preprocess_dataset(df, column_mapping)
        
        # Check column renaming
        assert "id" in processed_df.columns
        assert "question" in processed_df.columns
        assert "ground_truth" in processed_df.columns
        
        # Check normalization of answers
        assert "normalized_extracted" in processed_df.columns
        assert "normalized_ground_truth" in processed_df.columns
        
        # Check specific values
        assert processed_df.loc[2, "normalized_ground_truth"] == "180"
        assert processed_df.loc[2, "normalized_extracted"] == "180"


class TestDataset:
    """Tests for Dataset class"""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing"""
        return Config(base_config={
            "dataset": {
                "name": "test_dataset",
                "base_path": "data",
                "raw_file": "test_data.csv",
                "columns": {
                    "id": "ID",
                    "problem": "Problem",
                    "solution": "Solution",
                    "answer": "Answer",
                    "reasoning_trace": "model_completion",
                    "extracted_answer": "extracted_answer"
                },
                "train_test_split": 0.8,
                "random_seed": 42
            }
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataset(self, temp_dir, sample_config, monkeypatch):
        """Create a sample dataset file and configure Dataset to use it"""
        # Create data directory
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create test CSV file
        test_file = data_dir / "test_data.csv"
        df = pd.DataFrame(SAMPLE_DATA)
        df.to_csv(test_file, index=False)
        
        # Directly override the config's paths to use our temp directory
        config_dict = sample_config.to_dict()
        config_dict["dataset"]["base_path"] = str(data_dir)
        new_config = Config(base_config=config_dict)
        
        # Return dataset instance
        return Dataset(new_config)
    
    def test_dataset_initialization(self, sample_config):
        """Test Dataset initialization"""
        dataset = Dataset(sample_config)
        
        assert dataset.name == "test_dataset"
        assert dataset.train_test_split == 0.8
        assert dataset.random_seed == 42
    
    def test_load_raw(self, sample_dataset):
        """Test loading raw data"""
        df = sample_dataset.load_raw()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["ID", "Problem", "Solution", "Answer", "model_completion", "extracted_answer"]
    
    def test_load_processed(self, sample_dataset):
        """Test loading and processing data"""
        df = sample_dataset.load_processed()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "id" in df.columns
        assert "question" in df.columns
        assert "normalized_extracted" in df.columns
    
    def test_split_train_test(self, sample_dataset):
        """Test train/test splitting"""
        train_df, test_df = sample_dataset.split_train_test(test_size=0.33)
        
        assert len(train_df) == 2  # 3 * 0.67 rounded
        assert len(test_df) == 1   # 3 * 0.33 rounded
    
    def test_get_problems(self, sample_dataset):
        """Test getting problems"""
        # Load the data first
        sample_dataset.load_processed()
        
        # Test full dataset
        all_problems = sample_dataset.get_problems(split="all")
        assert len(all_problems) == 3
        assert "id" in all_problems[0]
        assert "question" in all_problems[0]
        
        # Test train/test split
        train_problems = sample_dataset.get_problems(split="train")
        test_problems = sample_dataset.get_problems(split="test")
        assert len(train_problems) + len(test_problems) == 3
    
    def test_get_problem_by_id(self, sample_dataset):
        """Test getting a specific problem by ID"""
        # Load the data first
        sample_dataset.load_processed()
        
        # Get a specific problem
        problem = sample_dataset.get_problem_by_id("2024-I-1")
        assert problem["id"] == "2024-I-1"
        assert "question" in problem
        assert "reasoning_trace" in problem
        
        # Test non-existent ID
        with pytest.raises(ValueError):
            sample_dataset.get_problem_by_id("non-existent-id")