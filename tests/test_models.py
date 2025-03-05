"""
Tests for model interfaces and implementations.
"""
import pytest
import os
from unittest.mock import patch, MagicMock
import json

from src.utils.config import Config
from src.models.base import Model, ModelResponse
from src.models.deepseek import DeepSeekModel

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
        "timeout": 10,
        "retries": 2,
        "retry_delay": 1
    },
    "reasoning": {
        "think_tag": True,
        "max_extensions": 3,
        "target_token_count": 1000
    }
}

# Mock API responses
MOCK_COMPLETION_RESPONSE = {
    "id": "cmpl-12345",
    "object": "text_completion",
    "created": 1677858242,
    "model": "accounts/fireworks/models/deepseek-r1",
    "choices": [
        {
            "text": "This is a test completion",
            "index": 0,
            "logprobs": None,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 5,
        "total_tokens": 10
    }
}

MOCK_REASONING_RESPONSE = {
    "id": "cmpl-12346",
    "object": "text_completion",
    "created": 1677858243,
    "model": "accounts/fireworks/models/deepseek-r1",
    "choices": [
        {
            "text": "I'll solve this step by step. First, I need to understand the problem...",
            "index": 0,
            "logprobs": None,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 15,
        "total_tokens": 25
    }
}

MOCK_FINAL_ANSWER_RESPONSE = {
    "id": "cmpl-12347",
    "object": "text_completion",
    "created": 1677858244,
    "model": "accounts/fireworks/models/deepseek-r1",
    "choices": [
        {
            "text": "Therefore, the answer is \\boxed{42}",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 5,
        "total_tokens": 20
    }
}

class TestModelResponse:
    """Tests for the ModelResponse class"""
    
    def test_model_response_initialization(self):
        """Test basic initialization of ModelResponse"""
        response = ModelResponse(
            text="Test response",
            prompt="Test prompt",
            tokens_used=10,
            model_name="test-model",
            raw_response={"key": "value"}
        )
        
        assert response.text == "Test response"
        assert response.prompt == "Test prompt"
        assert response.tokens_used == 10
        assert response.model_name == "test-model"
        assert response.raw_response == {"key": "value"}
    
    def test_model_response_str(self):
        """Test string representation of ModelResponse"""
        response = ModelResponse(
            text="Test response",
            prompt="Test prompt"
        )
        
        assert str(response) == "Test response"
    
    def test_model_response_to_dict(self):
        """Test conversion of ModelResponse to dictionary"""
        response = ModelResponse(
            text="Test response",
            prompt="Test prompt",
            tokens_used=10,
            model_name="test-model"
        )
        
        response_dict = response.to_dict()
        assert response_dict["text"] == "Test response"
        assert response_dict["prompt"] == "Test prompt"
        assert response_dict["tokens_used"] == 10
        assert response_dict["model_name"] == "test-model"


class TestDeepSeekModel:
    """Tests for the DeepSeekModel class"""
    
    @pytest.fixture
    def config(self):
        """Create a configuration object for testing"""
        return Config(base_config=SAMPLE_CONFIG)
    
    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up environment variables for testing"""
        monkeypatch.setenv("FIREWORKS_API_KEY", "fake-api-key")
    
    def test_initialization(self, config, mock_env):
        """Test initialization of DeepSeekModel"""
        model = DeepSeekModel(config)
        
        assert model.model_id == "accounts/fireworks/models/deepseek-r1"
        assert model.max_tokens == 1000
        assert model.temperature == 0.7
        assert model.api_key == "fake-api-key"
        assert model.think_tag is True
        assert model.max_extensions == 3
        assert model.target_token_count == 1000
    
    def test_initialization_missing_api_key(self, config, monkeypatch):
        """Test initialization fails with missing API key"""
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="FIREWORKS_API_KEY not found"):
            DeepSeekModel(config)
    
    @patch('requests.post')
    def test_generate(self, mock_post, config, mock_env):
        """Test basic text generation"""
        # Configure the mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_COMPLETION_RESPONSE
        mock_post.return_value = mock_response
        
        # Create model and generate text
        model = DeepSeekModel(config)
        response = model.generate(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.5
        )
        
        # Check the API was called correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['json']['prompt'] == "Test prompt"
        assert kwargs['json']['max_tokens'] == 100
        assert kwargs['json']['temperature'] == 0.5
        
        # Check the response
        assert isinstance(response, ModelResponse)
        assert response.text == "This is a test completion"
        assert response.tokens_used == 10
    
    @patch('requests.post')
    def test_generate_reasoning(self, mock_post, config, mock_env):
        """Test generating reasoning trace"""
        # Configure the mocks for initial reasoning and final answer
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = MOCK_REASONING_RESPONSE
        
        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = MOCK_FINAL_ANSWER_RESPONSE
        
        mock_post.side_effect = [mock_response1, mock_response2]
        
        # Create model and generate reasoning
        model = DeepSeekModel(config)
        result = model.generate_reasoning(
            question="What is the answer to life, the universe, and everything?",
            max_extensions=1
        )
        
        # Check API was called correctly
        assert mock_post.call_count == 2
        
        # Check the result structure
        assert "question" in result
        assert "reasoning" in result
        assert "answer" in result
        assert result["answer"] == "Therefore, the answer is \\boxed{42}"
    
    @patch('requests.post')
    def test_extract_answer(self, mock_post, config, mock_env):
        """Test answer extraction from boxed format"""
        model = DeepSeekModel(config)
        
        # Test with boxed answer
        answer1 = model.extract_answer("The solution is \\boxed{42}")
        assert answer1 == "42"
        
        # Test with "Therefore" pattern
        answer2 = model.extract_answer("Therefore, the answer is 42.")
        assert answer2 == "42"
        
        # Test with no answer format
        answer3 = model.extract_answer("This text has no answer format")
        assert answer3 == ""
    
    @patch('requests.post')
    def test_summarize_reasoning(self, mock_post, config, mock_env):
        """Test summarizing reasoning trace"""
        # Configure the mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"text": "This is a summary of the reasoning."}],
            "usage": {"total_tokens": 5}
        }
        mock_post.return_value = mock_response
        
        # Create model and summarize
        model = DeepSeekModel(config)
        summary = model.summarize_reasoning(
            reasoning_trace="This is a long reasoning trace that needs to be summarized."
        )
        
        # Check API was called correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "Summarize the following reasoning trace" in kwargs['json']['prompt']
        
        # Check the result
        assert summary == "This is a summary of the reasoning."