"""
Test script to implement and test a continuation mechanism for the Fireworks Completions API.
This addresses the 8,192 token limit by seamlessly continuing generation when needed.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
import tiktoken
from typing import Dict, Any, List, Optional, Tuple, Iterator
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Get API key from environment
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY not found in environment")

# API endpoints
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

def format_chat_for_completions(messages: List[Dict[str, str]], tokenizer):
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

def call_completions_api(
    prompt: str,
    max_tokens: int = 8192,  # Set to known limit
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 40,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """Make a basic call to the Fireworks Completions API"""
    if verbose:
        print(f"\n=== Calling Completions API ===")
        print(f"Model: {MODEL_NAME}")
        print(f"Max tokens: {max_tokens}")
        print(f"Prompt tokens: {count_tokens(prompt)}")
        print(f"Prompt length (chars): {len(prompt)}")
    
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
    
    if verbose:
        print(f"Response time: {end_time - start_time:.2f} seconds")
    
    if response.status_code != 200:
        if verbose:
            print(f"Error: {response.status_code}")
            print(response.text)
        return {"error": f"HTTP {response.status_code}", "details": response.text}
    
    response_json = response.json()
    
    # Extract and print metrics if verbose
    if verbose and "usage" in response_json:
        usage = response_json["usage"]
        finish_reason = response_json["choices"][0]["finish_reason"] if "choices" in response_json and response_json["choices"] else "unknown"
        completion_tokens = usage.get("completion_tokens", 0)
        
        print("\n=== Response Metrics ===")
        print(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {usage.get('total_tokens', 0)}")
        print(f"Finish reason: {finish_reason}")
        print(f"Max tokens specified: {max_tokens}")
        print(f"Percentage of max used: {(completion_tokens / max_tokens) * 100:.2f}%")
        
        if finish_reason == "length":
            print("\n⚠️ Response was cut off due to token limit!")
            
        if "choices" in response_json and response_json["choices"]:
            content = response_json["choices"][0]["text"]
            print(f"Content length (chars): {len(content)}")
            print(f"Content tokens (approx): {count_tokens(content)}")
    
    return response_json

def create_continuation_prompt(original_messages: List[Dict[str, str]], generated_text: str, tokenizer) -> str:
    """
    Create a prompt for continuing generation from where it left off.
    
    Args:
        original_messages: The original chat messages
        generated_text: The text generated so far (that got cut off)
        tokenizer: HF tokenizer for chat formatting
        
    Returns:
        A properly formatted continuation prompt string
    """
    # 1. Keep the same system context if any
    system_message = next((msg for msg in original_messages if msg["role"] == "system"), None)
    
    # 2. Create a new prompt with the original question plus the generated text so far
    user_message = next((msg for msg in original_messages if msg["role"] == "user"), {"content": ""})
    
    # 3. Determine the context window size for the continuation
    # Keep approximately the last 1000-2000 tokens of generated text for continuity
    generated_tokens = count_tokens(generated_text)
    continuation_context_tokens = min(generated_tokens, 2000)  # Use all if less than 2000 tokens
    
    # Very rough approximation: 1 token ≈ 4 characters for English text
    char_estimate = continuation_context_tokens * 4
    continuation_context = generated_text[-char_estimate:] if len(generated_text) > char_estimate else generated_text
    
    # 4. Build new continuation messages
    continuation_messages = []
    if system_message:
        continuation_messages.append(system_message)
    
    # Combine the original question with a request to continue from previous output
    continuation_messages.append({
        "role": "user",
        "content": f"{user_message['content']}\n\nHere's what you've written so far, please continue from where you left off:\n\n{continuation_context}"
    })
    
    # 5. Format for completions API
    return format_chat_for_completions(continuation_messages, tokenizer)

def generate_with_continuation(
    messages: List[Dict[str, str]],
    tokenizer,
    temperature: float = 0.6,
    max_total_tokens: int = 24576,  # Target a larger final size, e.g., 24k tokens (3 continuations)
    max_continuations: int = 5,  # Prevent infinite loops
    save_results: bool = True
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a long completion with automatic continuation when the token limit is reached.
    
    Args:
        messages: List of chat messages (as dict with role and content)
        tokenizer: HuggingFace tokenizer for the model
        temperature: Generation temperature
        max_total_tokens: Target total generation length in tokens
        max_continuations: Maximum number of continuation attempts
        save_results: Whether to save intermediate results to files
        
    Returns:
        Tuple of (full_text, detailed_responses)
    """
    print(f"\n{'=' * 50}")
    print(f"STARTING GENERATION WITH CONTINUATION")
    print(f"{'=' * 50}")
    print(f"Target total tokens: {max_total_tokens}")
    print(f"Max continuations: {max_continuations}")
    
    # Step 1: Format the initial prompt
    formatted_prompt = format_chat_for_completions(messages, tokenizer)
    
    # Step 2: Start the first generation (limited to 8192 tokens)
    current_tokens = 0
    all_text = ""
    all_responses = []
    
    # Create output directory if saving results
    if save_results:
        output_dir = "continuation_results"
        os.makedirs(output_dir, exist_ok=True)
    
    # Track iterations
    iteration = 0
    
    while current_tokens < max_total_tokens and iteration < max_continuations:
        print(f"\n--- Continuation Iteration {iteration} ---")
        
        # Generate response with the single-call token limit (8192)
        response = call_completions_api(
            prompt=formatted_prompt,
            max_tokens=8192,  # Use the known limit
            temperature=temperature
        )
        
        # Store the full response
        all_responses.append(response)
        
        # Extract text from response
        if "choices" in response and response["choices"]:
            new_text = response["choices"][0]["text"]
            current_tokens += count_tokens(new_text)
            all_text += new_text
            
            # Save this iteration's results
            if save_results:
                with open(f"{output_dir}/iteration_{iteration}_response.json", "w") as f:
                    json.dump(response, f, indent=2)
                
                with open(f"{output_dir}/iteration_{iteration}_text.txt", "w") as f:
                    f.write(new_text)
                
                with open(f"{output_dir}/full_text_so_far.txt", "w") as f:
                    f.write(all_text)
            
            # Check if we need to continue
            finish_reason = response["choices"][0]["finish_reason"]
            if finish_reason != "length" or current_tokens >= max_total_tokens:
                print(f"\nGeneration complete: finish_reason={finish_reason}, total_tokens={current_tokens}")
                break
            
            # Create a continuation prompt for the next iteration
            formatted_prompt = create_continuation_prompt(messages, all_text, tokenizer)
            
            # Prepare for next iteration
            iteration += 1
            
            # Add a short pause between API calls
            time.sleep(1)
        else:
            print("Error: No completion in response")
            break
    
    print(f"\nGeneration with continuation complete:")
    print(f"Total tokens generated: {current_tokens}")
    print(f"Total iterations: {iteration + 1}")
    print(f"Full text saved to: {output_dir}/full_text_so_far.txt")
    
    return all_text, all_responses

def test_with_existing_response():
    """Test continuation using an existing API response file if available"""
    output_dir = "api_test_results"
    existing_file = f"{output_dir}/long_completions_max8192.json"
    
    if not os.path.exists(existing_file):
        print(f"Existing response file not found: {existing_file}")
        return None
    
    print(f"\n{'=' * 50}")
    print(f"TESTING CONTINUATION WITH EXISTING RESPONSE")
    print(f"{'=' * 50}")
    
    # Load the existing response
    with open(existing_file, "r") as f:
        response = json.load(f)
    
    # Extract the generated text
    if "choices" in response and response["choices"]:
        generated_text = response["choices"][0]["text"]
        print(f"Found existing response with {count_tokens(generated_text)} tokens")
        
        # Use this response as the starting point for continuation
        tokenizer = load_hf_tokenizer(HF_MODEL_NAME)
        if not tokenizer:
            print("Failed to load tokenizer. Exiting.")
            return None
        
        # Create the original messages
        original_messages = [{
            "role": "user", 
            "content": "Solve the following AIME problem. All answers are integers ranging from 0 to 999, inclusive. Report your answer in \\boxed{} format.\n\nPROBLEM:\nLet $ S $ be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex in $ S $ is an endpoint of exactly one of the 12 segments."
        }]
        
        # Create the continuation prompt
        continuation_prompt = create_continuation_prompt(original_messages, generated_text, tokenizer)
        
        # Call the API for continuation
        continuation_response = call_completions_api(
            prompt=continuation_prompt,
            max_tokens=8192,
            temperature=0.6
        )
        
        # Save the continuation response
        continuation_dir = "continuation_results"
        os.makedirs(continuation_dir, exist_ok=True)
        
        with open(f"{continuation_dir}/continuation_response.json", "w") as f:
            json.dump(continuation_response, f, indent=2)
        
        # Extract and save the continuation text
        if "choices" in continuation_response and continuation_response["choices"]:
            continuation_text = continuation_response["choices"][0]["text"]
            
            with open(f"{continuation_dir}/continuation_text.txt", "w") as f:
                f.write(continuation_text)
            
            # Combine original and continuation
            full_text = generated_text + continuation_text
            
            with open(f"{continuation_dir}/full_text.txt", "w") as f:
                f.write(full_text)
            
            print(f"Full text tokens: {count_tokens(full_text)}")
            print(f"Full text saved to: {continuation_dir}/full_text.txt")
            
            return full_text
    
    return None

def test_full_continuation_flow():
    """Test the full generate_with_continuation flow from scratch"""
    # Load tokenizer
    tokenizer = load_hf_tokenizer(HF_MODEL_NAME)
    if not tokenizer:
        print("Failed to load tokenizer. Exiting.")
        return
    
    # Create messages for the math problem
    messages = [{
        "role": "user", 
        "content": "Solve the following AIME problem. All answers are integers ranging from 0 to 999, inclusive. Report your answer in \\boxed{} format.\n\nPROBLEM:\nLet $ S $ be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex in $ S $ is an endpoint of exactly one of the 12 segments."
    }]
    
    # Generate with continuation for a target of ~24k tokens (3 times the single-call limit)
    full_text, _ = generate_with_continuation(
        messages=messages,
        tokenizer=tokenizer,
        temperature=0.6,
        max_total_tokens=24576,
        max_continuations=3
    )
    
    return full_text

def main():
    """Run the main test suite"""
    # First, try to use an existing response if available
    existing_text = test_with_existing_response()
    
    if not existing_text:
        # If no existing response or continuation failed, test the full flow
        test_full_continuation_flow()
    
    print("\nAll tests completed. Results are in the continuation_results directory.")

if __name__ == "__main__":
    main()
