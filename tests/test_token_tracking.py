import os
import sys
import json
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.fireworks_client import FireworksModelClient
from src.llm.together_client import TogetherModelClient
from src.llm.base_client import TokenUsage, CostInfo

# Load environment variables
load_dotenv()

def test_fireworks_token_tracking():
    """Test token counting and cost tracking for Fireworks client with Llama 3.1 70B Instruct."""
    print("\n===== Testing Fireworks Token Tracking =====")
    
    # Initialize client with Llama 3.1 70B Instruct model
    client = FireworksModelClient(model_name="accounts/fireworks/models/llama-v3p1-70b-instruct")
    
    # Print pricing information
    print(f"Pricing: ${client.input_price_per_million_tokens}/M input tokens, ${client.output_price_per_million_tokens}/M output tokens")
    
    # Generate completion with token usage tracking
    prompt = "What is the capital of France? Please answer in one sentence."
    print(f"Prompt: {prompt}")
    
    messages = [{"role": "user", "content": prompt}]
    response, token_usage, cost_info = client.generate_completion(
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        return_usage=True
    )
    
    # Print response
    content = response["choices"][0]["message"]["content"]
    print(f"Response: {content}")
    
    # Print token usage and cost
    print(f"Token usage: {token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion, {token_usage.total_tokens} total")
    print(f"Cost: ${cost_info.total_cost:.6f} (${cost_info.prompt_cost:.6f} prompt, ${cost_info.completion_cost:.6f} completion)")
    
    return token_usage, cost_info

def test_together_token_tracking():
    """Test token counting and cost tracking for Together client with Llama 3.1 70B Instruct."""
    print("\n===== Testing Together Token Tracking =====")
    
    # Initialize client with Llama 3.1 70B Instruct model
    client = TogetherModelClient(model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
    
    # Print pricing information
    print(f"Pricing: ${client.input_price_per_million_tokens}/M input tokens, ${client.output_price_per_million_tokens}/M output tokens")
    
    # Generate completion with token usage tracking
    prompt = "What is the capital of France? Please answer in one sentence."
    print(f"Prompt: {prompt}")
    
    messages = [{"role": "user", "content": prompt}]
    response, token_usage, cost_info = client.generate_completion(
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        return_usage=True
    )
    
    # Print response
    content = response["choices"][0]["message"]["content"]
    print(f"Response: {content}")
    
    # Print token usage and cost
    print(f"Token usage: {token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion, {token_usage.total_tokens} total")
    print(f"Cost: ${cost_info.total_cost:.6f} (${cost_info.prompt_cost:.6f} prompt, ${cost_info.completion_cost:.6f} completion)")
    
    return token_usage, cost_info

if __name__ == "__main__":
    # Check if API calls are enabled
    if os.getenv("ENABLE_API_CALLS") != "1":
        print("API calls disabled. Set ENABLE_API_CALLS=1 to run.")
        sys.exit(0)
    
    # Run tests
    fireworks_usage, fireworks_cost = test_fireworks_token_tracking()
    together_usage, together_cost = test_together_token_tracking()
    
    # Compare results
    print("\n===== Comparison =====")
    print(f"Fireworks: {fireworks_usage.total_tokens} tokens, ${fireworks_cost.total_cost:.6f}")
    print(f"Together: {together_usage.total_tokens} tokens, ${together_cost.total_cost:.6f}")
