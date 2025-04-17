"""
Script to test Fireworks Completions API vs. Chat API
This tests whether the completions API allows longer outputs and can be used as a
drop-in replacement for the chat API with properly formatted prompts.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
import tiktoken
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Get API key from environment
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY not found in environment")

# API endpoints
CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
COMPLETIONS_URL = "https://api.fireworks.ai/inference/v1/completions"

# Model configuration
MODEL_NAME = "accounts/vivek-vajipey-84a360/deployedModels/deepseek-r1-distill-qwen-14b-61e7dbf1"
# Original HF model name from which the deployed model was created
HF_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Configure headers
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {FIREWORKS_API_KEY}"
}

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a text string"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def load_hf_tokenizer(model_name: str):
    """Load the Hugging Face tokenizer for the specified model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Successfully loaded tokenizer from: {model_name}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def format_chat_for_completions(messages: list, tokenizer):
    """Format chat messages for the completions API using the model's chat template"""
    if not tokenizer:
        raise ValueError("Tokenizer is required to format chat messages")
    
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt
    except Exception as e:
        print(f"Error applying chat template: {e}")
        raise

def test_chat_api(
    messages: list,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 40,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0
) -> Dict[str, Any]:
    """Test the Fireworks Chat API"""
    print("\n=== Testing Chat API ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    
    payload = {
        "model": MODEL_NAME,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "messages": messages,
        "stream": False
    }
    
    start_time = time.time()
    response = requests.post(
        CHAT_URL, 
        headers=headers, 
        data=json.dumps(payload)
    )
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f} seconds")
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": f"HTTP {response.status_code}", "details": response.text}
    
    response_json = response.json()
    
    # Extract and print metrics
    if "usage" in response_json:
        usage = response_json["usage"]
        finish_reason = response_json["choices"][0]["finish_reason"] if "choices" in response_json and response_json["choices"] else "unknown"
        completion_tokens = usage.get("completion_tokens", 0)
        
        print("\n=== Chat API Response Metrics ===")
        print(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {usage.get('total_tokens', 0)}")
        print(f"Finish reason: {finish_reason}")
        print(f"Max tokens specified: {max_tokens}")
        print(f"Percentage of max used: {(completion_tokens / max_tokens) * 100:.2f}%")
        
        if "choices" in response_json and response_json["choices"]:
            content = response_json["choices"][0]["message"]["content"]
            print(f"Content length (chars): {len(content)}")
            print(f"Content tokens (approx): {count_tokens(content)}")
            
            # Save the first 80 chars for comparison
            print(f"Content preview: {content[:80]}...")
    
    return response_json

def test_completions_api(
    prompt: str,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 40,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0
) -> Dict[str, Any]:
    """Test the Fireworks Completions API"""
    print("\n=== Testing Completions API ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Prompt tokens: {count_tokens(prompt)}")
    
    payload = {
        "model": MODEL_NAME,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "prompt": prompt,
        "stream": False
    }
    
    start_time = time.time()
    response = requests.post(
        COMPLETIONS_URL, 
        headers=headers, 
        data=json.dumps(payload)
    )
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f} seconds")
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": f"HTTP {response.status_code}", "details": response.text}
    
    response_json = response.json()
    
    # Extract and print metrics
    if "usage" in response_json:
        usage = response_json["usage"]
        finish_reason = response_json["choices"][0]["finish_reason"] if "choices" in response_json and response_json["choices"] else "unknown"
        completion_tokens = usage.get("completion_tokens", 0)
        
        print("\n=== Completions API Response Metrics ===")
        print(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {usage.get('total_tokens', 0)}")
        print(f"Finish reason: {finish_reason}")
        print(f"Max tokens specified: {max_tokens}")
        print(f"Percentage of max used: {(completion_tokens / max_tokens) * 100:.2f}%")
        
        if "choices" in response_json and response_json["choices"]:
            content = response_json["choices"][0]["text"]
            print(f"Content length (chars): {len(content)}")
            print(f"Content tokens (approx): {count_tokens(content)}")
            
            # Save the first 80 chars for comparison
            print(f"Content preview: {content[:80]}...")
    
    return response_json

def run_comparison_test(test_name: str, messages: list, tokenizer, max_tokens: int = 16384):
    """Run a comparison test between Chat API and Completions API"""
    print(f"\n{'=' * 50}")
    print(f"RUNNING COMPARISON TEST: {test_name}")
    print(f"{'=' * 50}")
    
    # Create output directory
    output_dir = "api_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test Chat API
    chat_response = test_chat_api(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0  # Use deterministic mode for comparison
    )
    
    # Format messages for Completions API
    formatted_prompt = format_chat_for_completions(messages, tokenizer)
    print(f"\nFormatted prompt for completions API:")
    print(f"{formatted_prompt[:200]}...")
    print(f"Prompt length (chars): {len(formatted_prompt)}")
    print(f"Prompt tokens: {count_tokens(formatted_prompt)}")
    
    # Test Completions API
    completions_response = test_completions_api(
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        temperature=0.0  # Use deterministic mode for comparison
    )
    
    # Save responses to files
    with open(f"{output_dir}/{test_name}_chat_response.json", "w") as f:
        json.dump(chat_response, f, indent=2)
    
    with open(f"{output_dir}/{test_name}_completions_response.json", "w") as f:
        json.dump(completions_response, f, indent=2)
    
    # Extract content for comparison
    chat_content = ""
    if "choices" in chat_response and chat_response["choices"]:
        chat_content = chat_response["choices"][0]["message"]["content"]
        with open(f"{output_dir}/{test_name}_chat_content.txt", "w") as f:
            f.write(chat_content)
    
    completions_content = ""
    if "choices" in completions_response and completions_response["choices"]:
        completions_content = completions_response["choices"][0]["text"]
        with open(f"{output_dir}/{test_name}_completions_content.txt", "w") as f:
            f.write(completions_content)
    
    # Compare the first 100 characters to see if they're similar
    print("\n=== Content Comparison ===")
    print(f"Chat API first 100 chars: {chat_content[:100]}")
    print(f"Completions API first 100 chars: {completions_content[:100]}")
    
    return chat_response, completions_response

def test_long_completions(tokenizer, max_tokens: int = 16384):
    """Test if the completions API allows longer outputs than the chat API"""
    print(f"\n{'=' * 50}")
    print(f"TESTING LONG COMPLETIONS")
    print(f"{'=' * 50}")
    
    # Create a simple math problem that requires long reasoning
    messages = [{
        "role": "user", 
        "content": "Solve the following AIME problem. All answers are integers ranging from 0 to 999, inclusive. Report your answer in \\boxed{} format.\n\nPROBLEM:\nLet $ S $ be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex in $ S $ is an endpoint of exactly one of the 12 segments."
    }]
    
    # Format for completions API
    formatted_prompt = format_chat_for_completions(messages, tokenizer)
    
    # Test with different max_tokens values
    for max_tokens_value in [8192, 16384, 32768]:
        print(f"\n--- Testing completions API with max_tokens={max_tokens_value} ---")
        completions_response = test_completions_api(
            prompt=formatted_prompt,
            max_tokens=max_tokens_value,
            temperature=0.6  # Use same temperature as in your experiment
        )
        
        # Save response to file
        output_dir = "api_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/long_completions_max{max_tokens_value}.json", "w") as f:
            json.dump(completions_response, f, indent=2)
        
        # Extract and save content
        if "choices" in completions_response and completions_response["choices"]:
            content = completions_response["choices"][0]["text"]
            with open(f"{output_dir}/long_completions_max{max_tokens_value}_content.txt", "w") as f:
                f.write(content)
        
        # Add a pause between tests
        time.sleep(2)

def main():
    """Run the main test suite"""
    os.makedirs("api_test_results", exist_ok=True)
    
    # Load HF tokenizer for chat template formatting
    tokenizer = load_hf_tokenizer(HF_MODEL_NAME)
    if not tokenizer:
        print("Failed to load tokenizer. Exiting.")
        return
    
    # Simple test to verify API equivalence
    simple_messages = [{
        "role": "user", 
        "content": "What is 2+2?"
    }]
    
    run_comparison_test("simple_math", simple_messages, tokenizer, max_tokens=1024)
    
    # Test with more complex problem
    math_problem_messages = [{
        "role": "user", 
        "content": "Solve the following math problem step by step: If f(x) = x^2 + 3x + 2, find f(5)."
    }]
    
    run_comparison_test("complex_math", math_problem_messages, tokenizer, max_tokens=2048)
    
    # Test if completions API allows longer outputs
    test_long_completions(tokenizer)
    
    print("\nAll tests completed. Results are in the api_test_results directory.")

if __name__ == "__main__":
    main()
