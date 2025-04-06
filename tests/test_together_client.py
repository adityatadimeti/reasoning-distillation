import os
import sys
import pytest
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.together_client import TogetherModelClient
from src.llm.model_factory import create_model_client
from src.llm.base_client import TokenUsage, CostInfo

# Load environment variables
load_dotenv()

def test_together_client_initialization():
    """Test that the client initializes correctly."""
    client = TogetherModelClient(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    assert client.model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    assert client.api_key is not None

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_model_factory_together():
    """Test that the model factory correctly creates a Together client."""
    # Test explicit provider
    client = create_model_client(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 
        provider="together"
    )
    assert isinstance(client, TogetherModelClient)
    
    # Test auto-detection
    client = create_model_client("meta-llama/Meta-Llama-3-8B-Instruct")
    assert isinstance(client, TogetherModelClient)

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_deepseek_r1_distill_llama_70b():
    """Test a simple completion with DeepSeek R1 Distill Llama 70B."""
    client = TogetherModelClient(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    response, finish_reason = client.generate_response(
        "What is 2+2?",
        max_tokens=10,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Check that we got a non-empty response
    assert response is not None
    assert len(response) > 0
    print(f"DeepSeek R1 Distill Llama 70B Response: {response}")
    print(f"Finish reason: {finish_reason}")

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_deepseek_r1_distill_qwen_14b():
    """Test a simple completion with DeepSeek R1 Distill Qwen 14B."""
    client = TogetherModelClient(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    response, finish_reason = client.generate_response(
        "What is the capital of France?",
        max_tokens=10,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Check that we got a non-empty response
    assert response is not None
    assert len(response) > 0
    print(f"DeepSeek R1 Distill Qwen 14B Response: {response}")
    print(f"Finish reason: {finish_reason}")

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_deepseek_r1_distill_qwen_1_5b():
    """Test a simple completion with DeepSeek R1 Distill Qwen 1.5B."""
    client = TogetherModelClient(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    response, finish_reason = client.generate_response(
        "What is the color of the sky?",
        max_tokens=10,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Check that we got a non-empty response
    assert response is not None
    assert len(response) > 0
    print(f"DeepSeek R1 Distill Qwen 1.5B Response: {response}")
    print(f"Finish reason: {finish_reason}")

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_deepseek_r1_distill_llama_70b_free():
    """Test a simple completion with DeepSeek R1 Distill Llama 70B Free."""
    client = TogetherModelClient(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")
    response, finish_reason = client.generate_response(
        "What is 3+3?",
        max_tokens=10,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Check that we got a non-empty response
    assert response is not None
    assert len(response) > 0
    print(f"DeepSeek R1 Distill Llama 70B Free Response: {response}")
    print(f"Finish reason: {finish_reason}")

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_token_counting_and_cost():
    """Test token counting and cost tracking functionality."""
    client = TogetherModelClient(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")
    
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
    
    print(f"Token usage: {token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion, {token_usage.total_tokens} total")
    print(f"Cost: ${cost_info.total_cost:.6f} (${cost_info.prompt_cost:.6f} prompt, ${cost_info.completion_cost:.6f} completion)")

def run_all_tests():
    """Run all API tests sequentially."""
    print("Testing Together client initialization...")
    test_together_client_initialization()
    print("✅ Initialization test passed")
    
    print("\nTesting model factory...")
    test_model_factory_together()
    print("✅ Model factory test passed")
    
    print("\nTesting DeepSeek R1 Distill Llama 70B...")
    test_deepseek_r1_distill_llama_70b()
    
    print("\nTesting DeepSeek R1 Distill Qwen 14B...")
    test_deepseek_r1_distill_qwen_14b()
    
    print("\nTesting DeepSeek R1 Distill Qwen 1.5B...")
    test_deepseek_r1_distill_qwen_1_5b()
    
    print("\nTesting DeepSeek R1 Distill Llama 70B Free...")
    test_deepseek_r1_distill_llama_70b_free()
    
    print("\nTesting token counting and cost tracking...")
    test_token_counting_and_cost()
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    # If run directly, execute the API tests
    if os.getenv("ENABLE_API_CALLS") == "1":
        run_all_tests()
    else:
        print("API calls disabled. Set ENABLE_API_CALLS=1 to run.")
