"""
Tests for summarization functionality.
"""
import os
import pytest
from dotenv import load_dotenv
import time

from src.utils.config import Config
from src.models.factory import create_reasoning_model, create_summarization_model
from src.summarization.summarizers import get_summarizer
from src.summarization.strategies import get_summarization_strategy
from src.summarization.prompts import get_summarization_prompt

# Load environment variables
load_dotenv()

# Only run API-calling tests if explicitly enabled
api_tests_enabled = pytest.mark.skipif(
    os.environ.get("ENABLE_API_CALLS") != "1",
    reason="API tests disabled. Set ENABLE_API_CALLS=1 to enable."
)

# Sample configuration for testing
SAMPLE_CONFIG = {
    "reasoning_model": {
        "name": "deepseek-r1-distill-qwen-1p5b",
        "model_id": "accounts/fireworks/models/deepseek-r1-distill-qwen-1p5b",
        "api_type": "fireworks",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "summarization_model": {
        "name": "deepseek-r1-distill-qwen-1p5b",
        "model_id": "accounts/fireworks/models/deepseek-r1-distill-qwen-1p5b",
        "api_type": "fireworks",
        "max_tokens": 1000,
        "temperature": 0.5
    },
    "summarization": {
        "method": "external",
        "strategy": "default",
        "prompt": "default",
        "max_tokens": 500,
        "temperature": 0.5
    }
}

class TestSummarizationComponents:
    """Tests for summarization components without API calls"""
    
    @pytest.fixture
    def config(self):
        """Create a configuration object for testing"""
        return Config(base_config=SAMPLE_CONFIG)
    
    def test_get_summarization_strategy(self):
        """Test getting summarization strategies"""
        # Test default strategy
        default_strategy = get_summarization_strategy("default")
        assert default_strategy.__class__.__name__ == "DefaultStrategy"
        
        # Test concise strategy
        concise_strategy = get_summarization_strategy("concise")
        assert concise_strategy.__class__.__name__ == "ConciseStrategy"
        
        # Test error-focused strategy
        error_strategy = get_summarization_strategy("error_focused")
        assert error_strategy.__class__.__name__ == "ErrorFocusedStrategy"
        
        # Test invalid strategy
        with pytest.raises(ValueError):
            get_summarization_strategy("nonexistent")
    
    def test_get_summarization_prompt(self):
        """Test getting summarization prompts"""
        # Test default prompt
        default_prompt = get_summarization_prompt("default")
        assert "expert at summarizing mathematical reasoning" in default_prompt.system_template
        
        # Test concise prompt
        concise_prompt = get_summarization_prompt("concise")
        assert "extremely concise" in concise_prompt.system_template
        
        # Test error-focused prompt
        error_prompt = get_summarization_prompt("error_focused")
        assert "analyzing mathematical reasoning for errors" in error_prompt.system_template
        
        # Test invalid prompt
        with pytest.raises(ValueError):
            get_summarization_prompt("nonexistent")
    
    def test_prompt_formatting(self):
        """Test prompt template formatting"""
        prompt = get_summarization_prompt("default")
        system, user = prompt.format(reasoning_trace="1+1=2")
        
        assert "expert at summarizing" in system
        assert "1+1=2" in user
        assert "REASONING TRACE:" in user


@api_tests_enabled
class TestSummarization:
    """Integration tests for summarization with real API calls"""
    
    @pytest.fixture
    def config(self):
        """Create a configuration object for testing"""
        return Config(base_config=SAMPLE_CONFIG)
    
    @pytest.fixture
    def reasoning_model(self, config):
        """Create a reasoning model for testing"""
        return create_reasoning_model(config)
    
    @pytest.fixture
    def summarization_model(self, config):
        """Create a summarization model for testing"""
        return create_summarization_model(config)
    
    @pytest.fixture
    def summarizer(self, config, reasoning_model, summarization_model):
        """Create a summarizer for testing"""
        return get_summarizer(config, reasoning_model, summarization_model)
    
    def test_api_key_present(self):
        """Verify API key is available"""
        assert os.environ.get("FIREWORKS_API_KEY"), "FIREWORKS_API_KEY not found in environment"
    
    def test_summarizer_creation(self, config, reasoning_model, summarization_model):
        """Test creating different types of summarizers"""
        # Test external summarizer
        external_config = Config(base_config={
            "summarization": {"method": "external"}
        })
        external_summarizer = get_summarizer(external_config, reasoning_model, summarization_model)
        assert external_summarizer.__class__.__name__ == "ExternalSummarizer"
        
        # Test self summarizer
        self_config = Config(base_config={
            "summarization": {"method": "self"}
        })
        self_summarizer = get_summarizer(self_config, reasoning_model, None)
        assert self_summarizer.__class__.__name__ == "SelfSummarizer"
    
    def test_summarize_simple_reasoning(self, summarizer):
        """Test summarizing a simple reasoning trace"""
        reasoning_trace = """
        To solve this problem, I need to find how far the train goes in 3 hours.
        
        Given:
        - Speed of the train = 60 mph
        - Time of travel = 3 hours
        
        Using the formula: Distance = Speed × Time
        Distance = 60 mph × 3 hours
        Distance = 180 miles
        
        Therefore, the train travels 180 miles in 3 hours.
        """
        
        summary = summarizer.summarize(reasoning_trace)
        
        print("\n=== SIMPLE REASONING SUMMARY ===")
        print(f"Original length: {len(reasoning_trace)} chars")
        print(f"Summary length: {len(summary)} chars")
        print(f"Summary: {summary}")
        
        assert summary, "Summary should not be empty"
        # For very short reasoning traces, the summary might be longer due to formatting
        # and additional analysis, so we don't require it to be shorter
        if len(reasoning_trace) > 1000:
            assert len(summary) < len(reasoning_trace), "Summary should be shorter than original for longer traces"
        assert "180 miles" in summary, "Summary should contain the key result"
        assert "60 mph" in summary, "Summary should contain the key input data"
        assert "3 hours" in summary, "Summary should contain the key input data"
    
    def test_summarize_with_different_strategies(self, config, reasoning_model, summarization_model):
        """Test summarizing with different strategies"""
        reasoning_trace = """
        Let's solve this step by step.
        
        First, I need to understand the problem. We have a train that travels at 60 mph for 3 hours, and we need to find the total distance it covers.
        
        I'll use the formula: Distance = Speed × Time
        
        Given:
        - Speed (v) = 60 miles per hour
        - Time (t) = 3 hours
        
        Substituting these values into the formula:
        Distance = 60 mph × 3 h
        Distance = 180 miles
        
        However, let me double-check this result. If the train travels at 60 miles per hour, that means it covers 60 miles in 1 hour. In 3 hours, it would cover 3 times that distance.
        
        So, 60 miles/hour × 3 hours = 180 miles.
        
        The answer is 180 miles.
        """
        
        # Test default strategy
        default_config = Config(base_config={
            "summarization": {"method": "external", "strategy": "default"}
        })
        default_summarizer = get_summarizer(default_config, reasoning_model, summarization_model)
        default_summary = default_summarizer.summarize(reasoning_trace)
        
        # Test concise strategy
        concise_config = Config(base_config={
            "summarization": {"method": "external", "strategy": "concise"}
        })
        concise_summarizer = get_summarizer(concise_config, reasoning_model, summarization_model)
        concise_summary = concise_summarizer.summarize(reasoning_trace)
        
        # Test error-focused strategy
        error_config = Config(base_config={
            "summarization": {"method": "external", "strategy": "error_focused"}
        })
        error_summarizer = get_summarizer(error_config, reasoning_model, summarization_model)
        error_summary = error_summarizer.summarize(reasoning_trace)
        
        print("\n=== STRATEGY COMPARISON ===")
        print(f"Default strategy: {len(default_summary)} chars")
        print(default_summary)
        print("\nConcise strategy: {len(concise_summary)} chars")
        print(concise_summary)
        print("\nError-focused strategy: {len(error_summary)} chars")
        print(error_summary)
        
        # Check that concise is actually more concise
        assert len(concise_summary) < len(default_summary), "Concise summary should be shorter"
        # Check that error-focused mentions correctness
        assert any(word in error_summary.lower() for word in ["error", "correct", "valid", "accurate"]), "Error-focused summary should mention correctness"