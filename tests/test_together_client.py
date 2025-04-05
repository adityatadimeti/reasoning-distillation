import os
import sys
import pytest
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.together_client import TogetherModelClient
from src.llm.model_factory import create_model_client

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
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    # If run directly, execute the API tests
    if os.getenv("ENABLE_API_CALLS") == "1":
        run_all_tests()
    else:
        print("API calls disabled. Set ENABLE_API_CALLS=1 to run.")
