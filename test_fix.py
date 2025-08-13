#!/usr/bin/env python3
"""
Test the fix for Qwen3 reasoning content extraction
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

async def test_fix():
    """Test that the fix properly extracts reasoning content"""
    
    # Create client for reasoning server
    client = VLLMModelClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost",
        port=8007,
        max_model_len=32768
    )
    
    print("=" * 60)
    print("TESTING QWEN3 REASONING CONTENT FIX")
    print("=" * 60)
    
    # Test with reasoning mode
    content, finish_reason, token_usage, cost_info, api_calls = await client.generate_response_async(
        prompt="What is 15 + 27?",
        max_tokens=1000,
        temperature=0.6,
        qwen3_context="reasoning"
    )
    
    print(f"\n1. REASONING MODE RESULTS:")
    print(f"   Content length: {len(content)} chars")
    print(f"   Finish reason: {finish_reason}")
    print(f"   First 200 chars: {repr(content[:200])}")
    
    # Check if reasoning content is included
    has_reasoning = len(content) > 200  # Should be much longer now
    print(f"   Has detailed reasoning: {has_reasoning}")
    
    # Test with summarization mode
    content2, finish_reason2, token_usage2, cost_info2, api_calls2 = await client.generate_response_async(
        prompt="What is 15 + 27?",
        max_tokens=1000,
        temperature=0.6,
        qwen3_context="summarization"
    )
    
    print(f"\n2. SUMMARIZATION MODE RESULTS:")
    print(f"   Content length: {len(content2)} chars")
    print(f"   Finish reason: {finish_reason2}")
    print(f"   First 200 chars: {repr(content2[:200])}")
    
    # Compare lengths
    print(f"\n3. COMPARISON:")
    print(f"   Reasoning content: {len(content)} chars")
    print(f"   Summarization content: {len(content2)} chars")
    print(f"   Reasoning is {len(content) / len(content2):.1f}x longer")
    
    if len(content) > len(content2) * 2:
        print("   ✅ SUCCESS: Reasoning mode now includes detailed thinking!")
    else:
        print("   ❌ ISSUE: Reasoning content still not properly extracted")

if __name__ == "__main__":
    asyncio.run(test_fix())