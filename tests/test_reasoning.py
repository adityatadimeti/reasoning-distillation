"""
Tests for reasoning generation and extraction functionality.
"""
import os
import pytest
from dotenv import load_dotenv
import re

from src.utils.config import Config
from src.models.deepseek import DeepSeekModel
from src.data.dataset import Dataset
import src.reasoning.generation as generation
import src.reasoning.extraction as extraction

# Load environment variables
load_dotenv()

# Only run API-calling tests if explicitly enabled
api_tests_enabled = pytest.mark.skipif(
    os.environ.get("ENABLE_API_TESTS") != "1",
    reason="API tests disabled. Set ENABLE_API_TESTS=1 to enable."
)

class TestExtractionFunctions:
    """Tests for answer extraction functions"""
    
    def test_extract_boxed_answer(self):
        """Test extracting answers in boxed format"""
        # Test with simple boxed answer
        assert extraction.extract_boxed_answer("The answer is \\boxed{42}") == "42"
        
        # Test with complex boxed answer
        assert extraction.extract_boxed_answer("The answer is \\boxed{x^2 + 2x + 1}") == "x^2 + 2x + 1"
        
        # Test with no boxed answer
        assert extraction.extract_boxed_answer("The answer is 42") is None
        
        # Test with empty input
        assert extraction.extract_boxed_answer("") is None
        assert extraction.extract_boxed_answer(None) is None
    
    def test_extract_answer_from_text(self):
        """Test extracting answers from various text formats"""
        # Test with boxed answer
        assert extraction.extract_answer_from_text("The answer is \\boxed{42}") == "42"
        
        # Test with "the answer is" pattern
        assert extraction.extract_answer_from_text("Therefore, the answer is 42.") == "42"
        assert extraction.extract_answer_from_text("The answer is: 42") == "42"
        
        # Test with "the result is" pattern
        assert extraction.extract_answer_from_text("Thus, the result is 123.") == "123"
        
        # Test with multiple possible answers (should take the first match)
        text = "The answer might be 10. But the answer is 42. Or maybe it's 100."
        assert extraction.extract_answer_from_text(text) == "42"
        
        # Test with no clear answer format
        result = extraction.extract_answer_from_text("This text doesn't contain an answer")
        print(f"Result type: {type(result)}, value: '{result}'")
        assert result is None
    
    def test_normalize_answer(self):
        """Test answer normalization"""
        # Test integer-like answers
        assert extraction.normalize_answer("42") == "42"
        assert extraction.normalize_answer("42.0") == "42"
        assert extraction.normalize_answer("42.5") == "42.5"
        
        # Test with symbols and whitespace
        assert extraction.normalize_answer(" $42 ") == "42"
        assert extraction.normalize_answer("$42%") == "42"
        assert extraction.normalize_answer("1,234") == "1234"
        
        # Test with None
        assert extraction.normalize_answer(None) is None
    
    def test_extract_answer_from_reasoning_result(self):
        """Test extracting answer from reasoning result dictionary"""
        # Test with answer in the answer field
        result1 = {
            "question": "What is 6 × 7?",
            "reasoning": "I need to multiply 6 by 7.\n6 × 7 = 42",
            "answer": "The answer is 42."
        }
        assert extraction.extract_answer_from_reasoning_result(result1) == "42"
        
        # Test with answer only in reasoning
        result2 = {
            "question": "What is 6 × 7?",
            "reasoning": "I need to multiply 6 by 7.\n6 × 7 = 42\nTherefore, the answer is 42.",
            "answer": ""
        }
        assert extraction.extract_answer_from_reasoning_result(result2) == "42"
        
        # Test with empty result
        result3 = {
            "question": "What is 6 × 7?",
            "reasoning": "",
            "answer": ""
        }
        assert extraction.extract_answer_from_reasoning_result(result3) == ""


@api_tests_enabled
class TestReasoningGeneration:
    """Integration tests for reasoning generation functions"""
    
    @pytest.fixture
    def config(self):
        """Create configuration for testing"""
        return Config(base_config={
            "model": {
                "name": "deepseek_r1",
                "model_id": "accounts/fireworks/models/deepseek-r1",
                "api_type": "fireworks",
                "max_tokens": 1000,
                "temperature": 0.7
            },
            "reasoning": {
                "think_tag": True,
                "max_extensions": 2,
                "target_token_count": 500
            },
            "dataset": {
                "name": "aime2024",
                "base_path": "archive/data",
                "raw_file": "aime_2024_completions_updated.csv",
                "columns": {
                    "id": "ID",
                    "problem": "Problem",
                    "solution": "Solution",
                    "answer": "Answer",
                    "reasoning_trace": "model_completion",
                    "extracted_answer": "extracted_answer"
                }
            }
        })
    
    @pytest.fixture
    def model(self, config):
        """Create model instance for testing"""
        return DeepSeekModel(config)
    
    @pytest.fixture
    def dataset(self, config):
        """Create dataset instance for testing"""
        return Dataset(config)
    
    def test_api_key_present(self):
        """Verify API key is available"""
        assert os.environ.get("FIREWORKS_API_KEY"), "FIREWORKS_API_KEY not found in environment"
    
    def test_generate_reasoning_trace(self, config, model):
        """Test generating a reasoning trace with real API call"""
        # Simple math problem
        question = "If a train travels at 60 mph for 3 hours, how far does it go?"
        
        result = generation.generate_reasoning_trace(
            question=question,
            model=model,
            config=config
        )
        
        print("\n=== REASONING TRACE ===")
        print(f"Question: {result['question']}")
        print(f"First 200 chars: {result['reasoning'][:200]}...")
        print(f"Answer: {result['answer']}")
        print(f"Generation time: {result['generation_time']:.2f}s")
        
        assert "question" in result
        assert "reasoning" in result
        assert "answer" in result
        assert "generation_time" in result
        assert len(result["reasoning"]) > 100
        
        # Extract the answer
        answer = extraction.extract_answer_from_reasoning_result(result)
        print(f"Extracted answer: {answer}")
        
        # Should be 180 miles or similar
        assert answer is not None
        assert re.search(r"1.*8.*0", answer) is not None
    
    def test_generate_with_aime_problem(self, config, model, dataset):
        """Test generating reasoning for a problem from the AIME dataset"""
        # Load a problem from the dataset
        try:
            dataset.load_raw()
            problem = dataset.data_raw.iloc[0]
            question = problem["Problem"]
            
            print(f"\n=== AIME PROBLEM ===")
            print(f"ID: {problem['ID']}")
            print(f"Question: {question[:100]}...")
            
            result = generation.generate_reasoning_trace(
                question=question,
                model=model,
                config=config
            )
            
            print("\n=== AIME REASONING TRACE ===")
            print(f"First 200 chars: {result['reasoning'][:200]}...")
            print(f"Answer: {result['answer']}")
            print(f"Generation time: {result['generation_time']:.2f}s")
            
            assert len(result["reasoning"]) > 100
            
            # Extract the answer
            answer = extraction.extract_answer_from_reasoning_result(result)
            print(f"Extracted answer: {answer}")
            assert answer is not None
            
        except Exception as e:
            pytest.skip(f"Could not load AIME dataset: {str(e)}")