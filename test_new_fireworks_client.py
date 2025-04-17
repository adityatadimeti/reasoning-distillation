"""
Test script for the revised FireworksModelClient using completions API with continuations.
"""

import os
import json
import asyncio
import argparse
from dotenv import load_dotenv

from src.llm.fireworks_client import FireworksModelClient
from src.llm.tokenization import count_tokens

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def verify_api_key():
    """Verify the Fireworks API key is properly configured."""
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("ERROR: FIREWORKS_API_KEY environment variable not found!")
        return False
    
    # Print the first and last few characters of the key for debugging
    if len(api_key) > 10:
        print(f"API key found: {api_key[:5]}...{api_key[-5:]} (length: {len(api_key)})")
        return True
    else:
        print(f"API key found but it seems too short: {len(api_key)} characters")
        return False

async def test_basic_generation(model_name: str):
    """Test basic generation without continuation."""
    print(f"\n{'=' * 60}")
    print(f"TESTING BASIC GENERATION (NO CONTINUATION)")
    print(f"{'=' * 60}")
    
    # Create the client with explicit verbose flag
    client = FireworksModelClient(model_name=model_name)
    prompt = "What is 2+2?"
    
    print(f"Prompt: {prompt}")
    print(f"Using model: {model_name}")
    
    try:
        print("Starting API call...")
        result = await asyncio.wait_for(
            client.generate_response_async(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.0,  # Use deterministic mode for testing
                verbose=True,
                enable_continuation=False  # Disable continuation for this test
            ),
            timeout=90  # Timeout after 90 seconds
        )
        print("API call completed successfully!")
    except asyncio.TimeoutError:
        print("ERROR: API call timed out after 90 seconds!")
        return "TIMEOUT"
    except Exception as e:
        print(f"ERROR: API call failed with exception: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"
    
    text, finish_reason, token_usage, cost_info = result
    
    print(f"\nFinish reason: {finish_reason}")
    print(f"Token usage: {token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion")
    print(f"Content length (chars): {len(text)}")
    print(f"Content tokens (approx): {count_tokens(text)}")
    print(f"First 200 chars:\n{text[:200]}...")
    
    # Save results
    os.makedirs("client_test_results", exist_ok=True)
    with open("client_test_results/basic_generation.txt", "w") as f:
        f.write(text)
    
    return text

async def test_continuation(model_name: str):
    """Test generation with continuation for a longer problem."""
    print(f"\n{'=' * 60}")
    print(f"TESTING GENERATION WITH CONTINUATION")
    print(f"{'=' * 60}")
    
    client = FireworksModelClient(model_name=model_name)
    
    # Use the AIME problem that previously hit the token limit
    prompt = """Solve the following AIME problem. All answers are integers ranging from 0 to 999, inclusive. Report your answer in \\boxed{} format.

PROBLEM:
Let $ S $ be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex in $ S $ is an endpoint of exactly one of the 12 segments."""
    
    print(f"Prompt: {prompt}")
    
    result = await client.generate_response_async(
        prompt=prompt,
        max_tokens=8192,  # Use the per-request limit
        temperature=0.6,  # Use the same temperature as in experiments
        top_p=0.95,
        top_k=40,
        verbose=True,
        enable_continuation=True,
        max_total_tokens=24576,  # Target up to ~24k tokens (3x the single request limit)
        max_continuations=3
    )
    
    text, finish_reason, token_usage, cost_info = result
    
    print(f"\nFinish reason: {finish_reason}")
    print(f"Token usage: {token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion, {token_usage.total_tokens} total")
    print(f"Content length (chars): {len(text)}")
    print(f"Content tokens (approx): {count_tokens(text)}")
    
    # Save results
    os.makedirs("client_test_results", exist_ok=True)
    with open("client_test_results/continuation_test.txt", "w") as f:
        f.write(text)
    
    # Check if the response contains a boxed answer
    if "\\boxed{" in text:
        print("\nFound boxed answer in the response!")
        # Extract the answer
        try:
            start_idx = text.find("\\boxed{") + 7
            end_idx = text.find("}", start_idx)
            if start_idx > 0 and end_idx > start_idx:
                answer = text[start_idx:end_idx]
                print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error extracting answer: {e}")
    else:
        print("\nNo boxed answer found in the response.")
    
    return text

async def main():
    parser = argparse.ArgumentParser(description="Test the FireworksModelClient with continuations")
    parser.add_argument("--model", default="accounts/vivek-vajipey-84a360/deployedModels/deepseek-r1-distill-qwen-14b-61e7dbf1", 
                      help="Fireworks model name")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout in seconds for API calls")
    parser.add_argument("--skip-continuation", action="store_true", help="Skip the continuation test")
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load environment variables for API keys
    print("Loading environment variables...")
    load_dotenv()
    
    # Verify API key
    if not await verify_api_key():
        print("ERROR: API key verification failed. Please check your FIREWORKS_API_KEY environment variable.")
        return
    
    print(f"Testing with model: {args.model}")
    
    # Test basic generation
    basic_result = await test_basic_generation(args.model)
    
    # Only proceed to continuation test if the basic test passed and we're not skipping it
    if basic_result != "TIMEOUT" and basic_result != "ERROR" and not args.skip_continuation:
        # Test generation with continuation
        await test_continuation(args.model)
    elif args.skip_continuation:
        print("Skipping continuation test as requested.")
    else:
        print("Skipping continuation test due to basic test failure.")

if __name__ == "__main__":
    asyncio.run(main())
