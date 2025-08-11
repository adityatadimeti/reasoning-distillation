#!/usr/bin/env python3
"""
Test script to verify GPT-OSS-20B client integration and reasoning effort control
Similar to test_qwen3_thinking.py but for GPT-OSS
"""
import asyncio
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

async def test_gpt_oss_client_setup():
    """Test the exact same client setup as used in experiments"""
    
    print("=" * 60)
    print("TESTING GPT-OSS CLIENT SETUP")
    print("=" * 60)
    
    # Create clients exactly as experiments do
    reasoning_client = VLLMModelClient(
        model_name="openai/gpt-oss-20b",
        host="localhost", 
        port=8012,
        max_model_len=32768
    )
    
    summarization_client = VLLMModelClient(
        model_name="openai/gpt-oss-20b",
        host="localhost",
        port=8013, 
        max_model_len=32768
    )
    
    print(f"Reasoning client - Model name: {reasoning_client.model_name}")
    print(f"Reasoning client - Is GPT-OSS: {reasoning_client.is_gpt_oss}")
    print(f"Summarization client - Model name: {summarization_client.model_name}")  
    print(f"Summarization client - Is GPT-OSS: {summarization_client.is_gpt_oss}")
    
    test_prompt = "What is 15 + 27? Show your work step by step."
    
    # Test reasoning client (should show high effort reasoning)
    print("\n1. Testing REASONING client (gpt_oss_context='reasoning'):")
    print("-" * 50)
    
    try:
        content, finish_reason, usage, cost, api_calls = await reasoning_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=300,
            temperature=0.6,
            gpt_oss_context="reasoning"  # This should enable high effort
        )
        
        print(f"Response: {content}")
        print(f"Response length: {len(content)} characters")
        print(f"Contains detailed reasoning: {'step' in content.lower() or 'first' in content.lower()}")
        print(f"Finish reason: {finish_reason}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test summarization client (should show low effort responses)  
    print("\n2. Testing SUMMARIZATION client (gpt_oss_context='summarization'):")
    print("-" * 50)
    
    try:
        content, finish_reason, usage, cost, api_calls = await summarization_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=300, 
            temperature=0.6,
            gpt_oss_context="summarization"  # This should enable low effort
        )
        
        print(f"Response: {content}")
        print(f"Response length: {len(content)} characters")
        print(f"Contains detailed reasoning: {'step' in content.lower() or 'first' in content.lower()}")
        print(f"Finish reason: {finish_reason}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test summarization with a summary-like prompt 
    print("\n3. Testing SUMMARIZATION client with summary prompt:")
    print("-" * 50)
    
    summary_prompt = """Please provide a concise summary of this reasoning:
    
The problem asks to find 15 + 27. I need to add these two numbers together. 
I can break this down: 15 + 27. Starting with the ones place: 5 + 7 = 12. 
I write down 2 and carry the 1. Then the tens place: 1 + 2 = 3, plus the carried 1 = 4. 
So 15 + 27 = 42.

Provide a brief summary without extensive reasoning."""
    
    try:
        content, finish_reason, usage, cost, api_calls = await summarization_client.generate_response_async(
            prompt=summary_prompt,
            max_tokens=200,
            temperature=0.6, 
            gpt_oss_context="summarization"  # This should enable low effort
        )
        
        print(f"Response: {content}")
        print(f"Response length: {len(content)} characters")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test without gpt_oss_context (should use defaults)
    print("\n4. Testing WITHOUT gpt_oss_context (baseline):")
    print("-" * 50)
    
    try:
        content, finish_reason, usage, cost, api_calls = await reasoning_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=300,
            temperature=0.6
            # No gpt_oss_context parameter
        )
        
        print(f"Response: {content}")
        print(f"Response length: {len(content)} characters")
        print(f"Contains detailed reasoning: {'step' in content.lower() or 'first' in content.lower()}")
        print(f"Finish reason: {finish_reason}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("EXPECTED BEHAVIOR:")
    print("- Reasoning client: Should show detailed, step-by-step reasoning (high effort)")
    print("- Summarization client: Should show concise, direct responses (low effort)")
    print("- Summary prompt: Should produce brief summaries without extensive reasoning")
    print("- Baseline: Should use model defaults (likely medium effort)")
    print("=" * 60)

async def test_complex_reasoning_comparison():
    """Test with complex reasoning to compare high vs low effort"""
    
    print("\n" + "=" * 60)
    print("COMPLEX REASONING COMPARISON")
    print("=" * 60)
    
    # Create clients
    reasoning_client = VLLMModelClient(model_name="openai/gpt-oss-20b", host="localhost", port=8012)
    summarization_client = VLLMModelClient(model_name="openai/gpt-oss-20b", host="localhost", port=8013)
    
    complex_prompt = """
    A store sells apples for $2 each and oranges for $3 each. 
    If someone buys 5 fruits total and spends $12, how many apples and oranges did they buy?
    Solve this step by step.
    """
    
    # Test high effort (reasoning)
    print("\n1. HIGH EFFORT (reasoning context):")
    print("-" * 40)
    
    try:
        content, finish_reason, usage, cost, api_calls = await reasoning_client.generate_response_async(
            prompt=complex_prompt,
            max_tokens=400,
            temperature=0.1,
            gpt_oss_context="reasoning"
        )
        
        print(f"Response length: {len(content)} characters")
        print(f"Response: {content}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test low effort (summarization)  
    print("\n2. LOW EFFORT (summarization context):")
    print("-" * 40)
    
    try:
        content, finish_reason, usage, cost, api_calls = await summarization_client.generate_response_async(
            prompt=complex_prompt,
            max_tokens=400,
            temperature=0.1, 
            gpt_oss_context="summarization"
        )
        
        print(f"Response length: {len(content)} characters")
        print(f"Response: {content}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_gpt_oss_client_setup())
    asyncio.run(test_complex_reasoning_comparison())