import os
import sys
import pytest
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.fireworks_client import FireworksModelClient
from src.llm.base_client import TokenUsage, CostInfo

# Load environment variables
load_dotenv()

def test_fireworks_client_initialization():
    """Test that the client initializes correctly."""
    client = FireworksModelClient(model_name="accounts/vivek-vajipey-84a360/deployedModels/deepseek-r1-distill-qwen-14b-61e7dbf1")
    assert client.model_name == "accounts/vivek-vajipey-84a360/deployedModels/deepseek-r1-distill-qwen-14b-61e7dbf1"
    assert client.api_key is not None

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_simple_completion():
    """Test a simple completion to verify API connectivity."""
    client = FireworksModelClient(model_name="accounts/vivek-vajipey-84a360/deployedModels/deepseek-r1-distill-qwen-14b-61e7dbf1")
    response = client.generate_response(
        "What is 2+2?",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Check that we got a non-empty response
    assert response is not None
    assert len(response) > 0
    print(f"Response: {response}")

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_token_counting_and_cost():
    """Test token counting and cost tracking functionality."""
    client = FireworksModelClient(model_name="accounts/vivek-vajipey-84a360/deployedModels/deepseek-r1-distill-qwen-14b-61e7dbf1")
    
    # Ensure pricing is set
    assert client.input_price_per_million_tokens > 0
    assert client.output_price_per_million_tokens > 0
    
    # Test with return_usage=True
    response, token_usage, cost_info = client.generate_completion(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        return_usage=True
    )
    
    # Check response structure
    assert "choices" in response
    assert "usage" in response
    
    # Check token usage
    assert isinstance(token_usage, TokenUsage)
    assert token_usage.prompt_tokens > 0
    assert token_usage.completion_tokens > 0
    assert token_usage.total_tokens == token_usage.prompt_tokens + token_usage.completion_tokens
    
    # Check cost info
    assert isinstance(cost_info, CostInfo)
    assert cost_info.prompt_cost >= 0
    assert cost_info.completion_cost >= 0
    assert abs(cost_info.total_cost - (cost_info.prompt_cost + cost_info.completion_cost)) < 1e-10  # Account for floating point precision
    
    # Verify that the cost calculation is correct
    expected_prompt_cost = (token_usage.prompt_tokens / 1_000_000) * client.input_price_per_million_tokens
    expected_completion_cost = (token_usage.completion_tokens / 1_000_000) * client.output_price_per_million_tokens
    
    assert abs(cost_info.prompt_cost - expected_prompt_cost) < 1e-10
    assert abs(cost_info.completion_cost - expected_completion_cost) < 1e-10

if __name__ == "__main__":
    # If run directly, execute the API tests
    if os.getenv("ENABLE_API_CALLS") == "1":
        test_simple_completion()
        # test_token_counting_and_cost()
    else:
        print("API calls disabled. Set ENABLE_API_CALLS=1 to run.")