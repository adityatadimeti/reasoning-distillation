#!/usr/bin/env python3
"""
Debug script to examine the exact structure of vLLM responses for Qwen3 thinking mode
"""

import asyncio
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

async def debug_response_structure():
    """Debug the actual vLLM response structure"""
    
    # Create client for reasoning server (thinking enabled)
    client = VLLMModelClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost",
        port=8007,
        max_model_len=32768
    )
    
    messages = [{"role": "user", "content": "What is 2+2?"}]
    
    print("=" * 60)
    print("DEBUGGING VLLM RESPONSE STRUCTURE")
    print("=" * 60)
    
    # Get the raw response
    response, token_usage, cost_info = await client.generate_completion_async(
        messages=messages,
        max_tokens=200,
        temperature=0.6,
        qwen3_context="reasoning"
    )
    
    print("\n1. FULL RESPONSE STRUCTURE:")
    print("-" * 40)
    print(json.dumps(response, indent=2))
    
    print("\n2. CHOICES[0] STRUCTURE:")
    print("-" * 40)
    choice = response["choices"][0]
    print(json.dumps(choice, indent=2))
    
    print("\n3. MESSAGE STRUCTURE:")
    print("-" * 40)
    message = choice["message"]
    print(json.dumps(message, indent=2))
    
    print("\n4. AVAILABLE FIELDS IN MESSAGE:")
    print("-" * 40)
    for key in message.keys():
        print(f"- {key}: {type(message[key])}")
        if isinstance(message[key], str):
            print(f"  Length: {len(message[key])} chars")
            print(f"  First 100 chars: {repr(message[key][:100])}")
    
    print("\n5. CHECKING FOR REASONING CONTENT:")
    print("-" * 40)
    if "reasoning_content" in message:
        reasoning = message["reasoning_content"]
        print(f"Found reasoning_content: {len(reasoning)} chars")
        print(f"First 200 chars: {repr(reasoning[:200])}")
    else:
        print("No 'reasoning_content' field found")
        
    # Check if thinking content is in the main content field
    content = message["content"]
    has_think_start = "<think>" in content
    has_think_end = "</think>" in content
    print(f"\nContent analysis:")
    print(f"- Length: {len(content)} chars")
    print(f"- Has <think>: {has_think_start}")
    print(f"- Has </think>: {has_think_end}")
    
    if has_think_start:
        think_start = content.find("<think>")
        think_end = content.find("</think>")
        print(f"- <think> position: {think_start}")
        print(f"- </think> position: {think_end}")
        
        if think_end == -1:
            print("- WARNING: <think> tag not properly closed - content may be truncated!")

if __name__ == "__main__":
    asyncio.run(debug_response_structure())