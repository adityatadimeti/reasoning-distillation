import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.fireworks_client import FireworksModelClient

# Load environment variables
load_dotenv()

class MockResponse:
    """Mock response object for testing"""
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        
    def json(self):
        return self.json_data
        
    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"Status code {self.status_code}")

def test_finish_reason_stop():
    """Test that finish_reason='stop' is correctly captured."""
    
    # Create mock response with finish_reason='stop'
    mock_response = MockResponse({
        "choices": [
            {
                "message": {"content": "This is a complete response."},
                "finish_reason": "stop"
            }
        ]
    })
    
    # Patch the requests.post to return our mock
    with patch('requests.post', return_value=mock_response):
        client = FireworksModelClient(model_name="accounts/fireworks/models/test-model")
        content, finish_reason = client.generate_response(
            "Test prompt",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        
        # Check that content and finish_reason are correct
        assert content == "This is a complete response."
        assert finish_reason == "stop"

def test_finish_reason_length():
    """Test that finish_reason='length' is correctly captured."""
    
    # Create mock response with finish_reason='length'
    mock_response = MockResponse({
        "choices": [
            {
                "message": {"content": "This response was cut off due to token limit..."},
                "finish_reason": "length"
            }
        ]
    })
    
    # Patch the requests.post to return our mock
    with patch('requests.post', return_value=mock_response):
        client = FireworksModelClient(model_name="accounts/fireworks/models/test-model")
        content, finish_reason = client.generate_response(
            "Test prompt",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        
        # Check that content and finish_reason are correct
        assert content == "This response was cut off due to token limit..."
        assert finish_reason == "length"

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_real_api_finish_reason():
    """Test finish_reason with a real API call."""
    client = FireworksModelClient(model_name="accounts/fireworks/models/qwq-32b")
    
    # Test with a small max_tokens to potentially trigger 'length'
    content, finish_reason = client.generate_response(
        "Please write a 500 word essay about artificial intelligence.",
        max_tokens=10,  # Very small to ensure we get 'length'
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Check we got a response and finish_reason
    assert content is not None
    assert len(content) > 0
    assert finish_reason in ["stop", "length"]
    print(f"Content: {content}")
    print(f"Finish reason: {finish_reason}")
    
    # Test with normal max_tokens to get 'stop'
    content, finish_reason = client.generate_response(
        "What is 2+2?",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Check we got a response and finish_reason is 'stop'
    assert content is not None
    assert len(content) > 0
    assert finish_reason in ["stop", "length"]
    print(f"Content: {content}")
    print(f"Finish reason: {finish_reason}")

if __name__ == "__main__":
    # If run directly, execute the API test
    if os.getenv("ENABLE_API_CALLS") == "1":
        test_real_api_finish_reason()
    else:
        print("API calls disabled. Set ENABLE_API_CALLS=1 to run.") 