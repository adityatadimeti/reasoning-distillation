#!/usr/bin/env python3
"""
Test script to verify experiment setup matches the working API test
"""
import asyncio
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

async def test_experiment_setup():
    """Test the exact same client setup as used in experiments"""
    
    print("=" * 60)
    print("TESTING EXPERIMENT CLIENT SETUP")
    print("=" * 60)
    
    # Create clients exactly as experiments do
    reasoning_client = VLLMModelClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost", 
        port=8004,
        max_model_len=32768
    )
    
    summarization_client = VLLMModelClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost",
        port=8005, 
        max_model_len=32768
    )
    
    test_prompt = "What is 15 + 27? Show your work step by step."
    
    # Test reasoning client (should show <think> tags)
    print("\n1. Testing REASONING client (qwen3_context='reasoning'):")
    print("-" * 50)
    
    try:
        content, finish_reason, usage, cost, api_calls = await reasoning_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=300,
            temperature=0.6,
            qwen3_context="reasoning"  # This should enable thinking
        )
        
        print(f"Response: {content[:500]}...")
        print(f"Contains <think> tags: {'<think>' in content}")
        print(f"Contains </think> tags: {'</think>' in content}")
        print(f"Finish reason: {finish_reason}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test summarization client (should NOT show <think> tags)  
    print("\n2. Testing SUMMARIZATION client (qwen3_context='summarization'):")
    print("-" * 50)
    
    try:
        content, finish_reason, usage, cost, api_calls = await summarization_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=300, 
            temperature=0.6,
            qwen3_context="summarization"  # This should disable thinking
        )
        
        print(f"Response: {content[:500]}...")
        print(f"Contains <think> tags: {'<think>' in content}")
        print(f"Contains </think> tags: {'</think>' in content}")
        print(f"Finish reason: {finish_reason}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test summarization with a longer reasoning-like prompt 
    print("\n3. Testing SUMMARIZATION client with summary prompt:")
    print("-" * 50)
    
    summary_prompt = """Please provide a concise summary of this reasoning:
    
The problem asks to find 15 + 27. First I'll add the ones place: 5 + 7 = 12. I write down 2 and carry the 1. Then the tens place: 1 + 2 = 3, plus the carried 1 = 4. So 15 + 27 = 42.

Provide a brief summary without showing internal reasoning."""
    
    try:
        content, finish_reason, usage, cost, api_calls = await summarization_client.generate_response_async(
            prompt=summary_prompt,
            max_tokens=300,
            temperature=0.6, 
            qwen3_context="summarization"  # This should disable thinking
        )
        
        print(f"Response: {content}")
        print(f"Contains <think> tags: {'<think>' in content}")
        print(f"Contains </think> tags: {'</think>' in content}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("If summarization responses contain <think> tags, there's a client issue.")
    print("If they don't, the problem is in your experiment configuration.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_experiment_setup())