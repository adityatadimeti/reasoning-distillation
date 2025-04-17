"""
Script to test Fireworks API length limitations and stopping behavior.
This isolates the API call from the rest of the experiment framework.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
import tiktoken
from typing import Dict, Any, Optional, Tuple

# Load environment variables
load_dotenv()

# Get API key from environment
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY not found in environment")

# API endpoint
BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

# Test parameters
MODEL_NAME = "accounts/vivek-vajipey-84a360/deployedModels/deepseek-r1-distill-qwen-14b-61e7dbf1"

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

def test_api_with_params(
    prompt: str,
    max_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 40,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    verbose: bool = True
) -> Tuple[Dict[str, Any], int]:
    """
    Test the Fireworks API with specific parameters
    
    Returns:
        Tuple of (response_json, prompt_token_count)
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_token_count = count_tokens(prompt)
    
    if verbose:
        print(f"\n=== TEST CONFIGURATION ===")
        print(f"Model: {MODEL_NAME}")
        print(f"Prompt tokens: {prompt_token_count}")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Prompt length (chars): {len(prompt)}")
    
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
        BASE_URL, 
        headers=headers, 
        data=json.dumps(payload)
    )
    end_time = time.time()
    
    # Print response time
    if verbose:
        print(f"Response time: {end_time - start_time:.2f} seconds")
    
    # Check for errors
    if response.status_code != 200:
        if verbose:
            print(f"Error: {response.status_code}")
            print(response.text)
        return response.json() if response.content else {"error": f"HTTP {response.status_code}"}, prompt_token_count
    
    response_json = response.json()
    
    # Extract and print metrics
    if verbose and "usage" in response_json:
        usage = response_json["usage"]
        finish_reason = response_json["choices"][0]["finish_reason"] if "choices" in response_json and response_json["choices"] else "unknown"
        completion_tokens = usage.get("completion_tokens", 0)
        
        print("\n=== RESPONSE METRICS ===")
        print(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {usage.get('total_tokens', 0)}")
        print(f"Finish reason: {finish_reason}")
        print(f"Max tokens specified: {max_tokens}")
        print(f"Percentage of max used: {(completion_tokens / max_tokens) * 100:.2f}%")
        
        # Check if finish_reason is length but we're well below max_tokens
        if finish_reason == "length" and completion_tokens < (max_tokens * 0.9):
            print("\n⚠️ WARNING: Finish reason is 'length' but we're well below max_tokens!")
            print(f"  Used only {completion_tokens} out of {max_tokens} tokens ({(completion_tokens / max_tokens) * 100:.2f}%)")
    
    return response_json, prompt_token_count

def run_length_tests(base_prompt: str, test_name: str):
    """Run a series of tests with different max_tokens values"""
    print(f"\n{'=' * 50}")
    print(f"RUNNING TEST SUITE: {test_name}")
    print(f"{'=' * 50}")
    
    # Start with a modest token limit and increase
    for max_tokens in [4096, 8192, 16384, 32768, 65536]:
        print(f"\n--- Testing with max_tokens={max_tokens} ---")
        response, prompt_tokens = test_api_with_params(
            prompt=base_prompt,
            max_tokens=max_tokens
        )
        
        # Save full response to file for detailed analysis
        output_dir = "api_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/{test_name}_max{max_tokens}.json", "w") as f:
            json.dump(response, f, indent=2)
            
        content = ""
        if "choices" in response and response["choices"]:
            content = response["choices"][0]["message"]["content"]
            
            # Save content to file
            with open(f"{output_dir}/{test_name}_max{max_tokens}_content.txt", "w") as f:
                f.write(content)
        
        print(f"Full response saved to: {output_dir}/{test_name}_max{max_tokens}.json")
        print(f"Content saved to: {output_dir}/{test_name}_max{max_tokens}_content.txt")
        print(f"Content length (chars): {len(content)}")
        print(f"Content tokens (approx): {count_tokens(content)}")
        
        # Add a pause between tests
        time.sleep(2)

def test_with_problem_10():
    """Test with the specific problem that showed the issue"""
    prompt = """Solve the following AIME problem. All answers are integers ranging from 0 to 999, inclusive. Report your answer in \\boxed{} format.

PROBLEM:
Let $ S $ be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex in $ S $ is an endpoint of exactly one of the 12 segments.
"""
    run_length_tests(prompt, "problem_10")

def test_with_long_prompt():
    """Test with a longer prompt to see if prompt length affects completion token limit"""
    # Create a longer prompt by adding filler text
    base_prompt = """Solve the following AIME problem. All answers are integers ranging from 0 to 999, inclusive. Report your answer in \\boxed{} format.

PROBLEM:
Let $ S $ be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex in $ S $ is an endpoint of exactly one of the 12 segments.
"""
    # Add summary section to simulate your experiment's later iterations
    filler = """
Previous summary:
I approached this problem by considering the constraints on pairing vertices of a regular 24-gon. First, I recognized that we need to create 12 segments connecting all 24 vertices, with each vertex being an endpoint of exactly one segment. This is equivalent to finding the number of ways to partition 24 distinct objects into 12 unordered pairs.

The standard formula for this is (24!)/(12! × 2^12), which counts all possible pairings. However, this overcounts because we need segments of equal length. In a regular 24-gon, segments have equal length only when they connect vertices that are the same distance apart along the perimeter.

I identified that segments can connect vertices that are k positions apart, where k can be 1, 2, 3, ..., 12. Each k-value corresponds to a different segment length. For a given k, there are 24 possible segments (each vertex can form a segment with the vertex k positions away). Since we need 12 segments total, we must use segments of the same length, meaning we need to choose a single k-value and create 12 segments with that distance.

For a given k, we can create at most 12 segments (since there are 24 vertices total). For k=1, k=5, k=7, or k=11, we can create exactly 12 segments. For k=2 or k=10, we can create at most 8 segments. For k=3 or k=9, we can create at most 8 segments. For k=4 or k=8, we can create at most 6 segments. For k=6, we can create at most 4 segments.

Therefore, we have 4 values of k that allow us to create exactly 12 segments. For each of these k-values, there is only one way to arrange the 12 segments.

In total, there are 4 ways to draw the 12 segments of equal length connecting all 24 vertices.

Now I realize the above is incorrect. Let me recalculate more carefully. 
"""
    # Repeat the filler to make it longer
    long_prompt = base_prompt + filler * 5
    run_length_tests(long_prompt, "long_prompt")

def test_with_zero_temperature():
    """Test with temperature=0 to see if that affects stopping behavior"""
    prompt = """Solve the following AIME problem. All answers are integers ranging from 0 to 999, inclusive. Report your answer in \\boxed{} format.

PROBLEM:
Let $ S $ be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex in $ S $ is an endpoint of exactly one of the 12 segments.
"""
    print("\n=== TESTING WITH TEMPERATURE=0 ===")
    response, prompt_tokens = test_api_with_params(
        prompt=prompt,
        max_tokens=16384,
        temperature=0.0
    )
    
    # Save full response to file for detailed analysis
    output_dir = "api_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/temp0_test.json", "w") as f:
        json.dump(response, f, indent=2)
        
    if "choices" in response and response["choices"]:
        content = response["choices"][0]["message"]["content"]
        with open(f"{output_dir}/temp0_test_content.txt", "w") as f:
            f.write(content)

def main():
    """Run the main test suite"""
    os.makedirs("api_test_results", exist_ok=True)
    
    # Run the test with problem 10
    test_with_problem_10()
    
    # Run the test with a long prompt
    test_with_long_prompt()
    
    # Run the test with temperature=0
    test_with_zero_temperature()
    
    print("\nAll tests completed. Results are in the api_test_results directory.")

if __name__ == "__main__":
    main()
