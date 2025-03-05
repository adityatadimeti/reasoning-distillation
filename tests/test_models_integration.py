"""
Integration tests for model implementations using real API calls.
Only runs when explicitly enabled with ENABLE_API_TESTS=1 environment variable.
"""
import os
import pytest
import time
from dotenv import load_dotenv

from src.utils.config import Config
from src.models.deepseek import DeepSeekModel

# Load environment variables from .env file
load_dotenv()

# Skip all tests if integration tests are not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("ENABLE_API_TESTS") != "1",
    reason="Integration tests disabled. Set ENABLE_API_TESTS=1 to enable."
)

# Sample configuration for testing
SAMPLE_CONFIG = {
    "model": {
        "name": "deepseek_r1",
        "model_id": "accounts/fireworks/models/deepseek-r1",
        "api_type": "fireworks",
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 40,
        "timeout": 30,
        "retries": 2,
        "retry_delay": 1
    },
    "reasoning": {
        "think_tag": True,
        "max_extensions": 3,
        "target_token_count": 1000
    }
}

class TestDeepSeekModelIntegration:
    """Integration tests for DeepSeekModel with real API calls"""
    
    @pytest.fixture
    def config(self):
        """Create a configuration object for testing"""
        return Config(base_config=SAMPLE_CONFIG)
    
    def test_api_key_present(self):
        """Verify API key is available"""
        assert os.environ.get("FIREWORKS_API_KEY"), "FIREWORKS_API_KEY not found in environment"
    
    def test_basic_generation(self, config):
        """Test basic text generation with real API call"""
        model = DeepSeekModel(config)
        
        response = model.generate(
            prompt="Write a haiku about programming.",
            max_tokens=50,
            temperature=0.7
        )
        
        # Log response for inspection
        print("\n=== BASIC GENERATION RESPONSE ===")
        print(f"Text: {response.text}")
        print(f"Tokens used: {response.tokens_used}")
        print(f"Raw response: {response.raw_response}")
        
        assert response.text, "Response text should not be empty"
        assert response.tokens_used, "Token count should be available"
        assert isinstance(response.tokens_used, int), "Token count should be an integer"
    
    def test_reasoning_generation(self, config):
        """Test reasoning trace generation with real API call"""
        model = DeepSeekModel(config)
        
        # Simple math problem
        question = "If a train travels at 60 mph for 3 hours, how far does it go?"
        
        result = model.generate_reasoning(
            question=question,
            max_extensions=1,
            target_token_count=500
        )
        
        # Log response for inspection
        print("\n=== REASONING GENERATION RESPONSE ===")
        print(f"Question: {result['question']}")
        print(f"Reasoning (first 200 chars): {result['reasoning'][:200]}...")
        print(f"Answer: {result['answer']}")
        print(f"Extensions: {result['extensions']}")
        print(f"Token count: {result['estimated_token_count']}")
        
        assert result["reasoning"], "Reasoning should not be empty"
        assert result["answer"], "Answer should not be empty"
        
        # Test answer extraction
        extracted = model.extract_answer(result["answer"])
        print(f"Extracted answer: {extracted}")
        assert extracted, "Should be able to extract an answer"
    
    def test_summarization(self, config):
        """Test summarization with real API call"""
        model = DeepSeekModel(config)
        
        reasoning = """
        To solve this problem, I need to find how far the train goes in 3 hours.
        
        Given:
        - Speed of the train = 60 mph
        - Time of travel = 3 hours
        
        Using the formula: Distance = Speed x Time
        Distance = 60 mph x 3 hours
        Distance = 180 miles
        
        Therefore, the train travels 180 miles in 3 hours.
        """
        
        summary = model.summarize_reasoning(reasoning)
        
        # Log response for inspection
        print("\n=== SUMMARIZATION RESPONSE ===")
        print(f"Original reasoning length: {len(reasoning)} chars")
        print(f"Summary length: {len(summary)} chars")
        print(f"Summary: {summary}")
        
        assert summary, "Summary should not be empty"
        assert len(summary) < len(reasoning), "Summary should be shorter than original reasoning"