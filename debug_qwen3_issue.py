#!/usr/bin/env python3
"""
Debug script to test Qwen3 thinking mode and identify why reasoning is minimal.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

async def test_qwen3_thinking():
    """Test Qwen3 with and without thinking mode to debug the reasoning issue."""
    
    # Create Qwen3 client using the same config as the experiment
    client = VLLMModelClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost",
        port=8005,
        max_model_len=32768
    )
    
    # Test problem from the results
    test_problem = "Using all numbers from [95, 89, 47, 24], create an equation that equals 17.\n\nRules:\n- You must use every single number in your answer and each number can only be used once (i.e. if input numbers are [1, 2, 3, 4], then your answer must be composed of 1,2,3,4 where each number only appears once.)\n- Available operations: addition (+), subtraction (-), multiplication (*), division (/)\n- Use parentheses to control order of operations\n\n- For each list of input numbers, and a given target number, you are guaranteed that there exists a valid solution using the given rules.\n\nReturn your final equation in <answer> </answer> tags, for example if your answer is (1 + 2) * 3 - 4, then return <answer> (1 + 2) * 3 - 4 </answer>.\n\n"
    
    print("=" * 80)
    print("DEBUGGING QWEN3 THINKING MODE ISSUE")
    print("=" * 80)
    
    print("\n1. Testing WITHOUT thinking mode (qwen3_context=None):")
    print("-" * 50)
    
    try:
        response1, finish_reason1, usage1, cost1, calls1 = await client.generate_response_async(
            prompt=test_problem,
            max_tokens=27000,
            temperature=0.6,
            qwen3_context=None  # No thinking mode
        )
        
        print(f"Response length: {len(response1)} characters")
        print(f"Finish reason: {finish_reason1}")
        print(f"Token usage: {usage1.completion_tokens} completion tokens")
        print(f"First 200 chars: {response1[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n2. Testing WITH thinking mode (qwen3_context='reasoning'):")
    print("-" * 50)
    
    try:
        response2, finish_reason2, usage2, cost2, calls2 = await client.generate_response_async(
            prompt=test_problem,
            max_tokens=27000,
            temperature=0.6,
            qwen3_context="reasoning"  # Enable thinking mode
        )
        
        print(f"Response length: {len(response2)} characters")
        print(f"Finish reason: {finish_reason2}")
        print(f"Token usage: {usage2.completion_tokens} completion tokens")
        print(f"First 200 chars: {response2[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n3. Testing with a direct prompt that asks for reasoning:")
    print("-" * 50)
    
    reasoning_prompt = f"""Please solve this problem step by step, showing all your work and reasoning:

{test_problem}

Please think through this carefully and show your work."""
    
    try:
        response3, finish_reason3, usage3, cost3, calls3 = await client.generate_response_async(
            prompt=reasoning_prompt,
            max_tokens=27000,
            temperature=0.6,
            qwen3_context="reasoning"  # Enable thinking mode
        )
        
        print(f"Response length: {len(response3)} characters")
        print(f"Finish reason: {finish_reason3}")
        print(f"Token usage: {usage3.completion_tokens} completion tokens")
        print(f"First 200 chars: {response3[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    
    # Compare responses
    print(f"No thinking mode: {len(response1) if 'response1' in locals() else 0} chars")
    print(f"With thinking mode: {len(response2) if 'response2' in locals() else 0} chars")
    print(f"With explicit reasoning prompt: {len(response3) if 'response3' in locals() else 0} chars")

if __name__ == "__main__":
    asyncio.run(test_qwen3_thinking())