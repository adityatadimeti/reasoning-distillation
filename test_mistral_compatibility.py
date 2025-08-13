#!/usr/bin/env python3
"""
Test that the fix doesn't break Mistral/Magistral models
"""

import asyncio
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

async def test_mistral_compatibility():
    """Test that Mistral/Magistral models still work correctly after the fix"""
    
    # Test Magistral (reasoning server)
    magistral_client = VLLMModelClient(
        model_name="mistralai/Magistral-Small-2506",
        host="localhost",
        port=8000,  # Assuming this is your Magistral port
        max_model_len=40960
    )
    
    print("=" * 60)
    print("TESTING MISTRAL/MAGISTRAL COMPATIBILITY")
    print("=" * 60)
    
    try:
        # Test Magistral reasoning
        content, finish_reason, token_usage, cost_info, api_calls = await magistral_client.generate_response_async(
            prompt="What is 15 + 27? Show your work.",
            max_tokens=500,
            temperature=0.6
        )
        
        print(f"\n1. MAGISTRAL RESULTS:")
        print(f"   Content length: {len(content)} chars")
        print(f"   Finish reason: {finish_reason}")
        print(f"   First 200 chars: {repr(content[:200])}")
        print(f"   ✅ Magistral working correctly")
        
    except Exception as e:
        print(f"   ❌ Magistral test failed: {e}")
    
    # Now let's check what a Magistral response structure looks like
    print(f"\n2. CHECKING MAGISTRAL RESPONSE STRUCTURE:")
    try:
        # Get raw response to check structure
        response, token_usage, cost_info = await magistral_client.generate_completion_async(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=200,
            temperature=0.6
        )
        
        message = response["choices"][0]["message"]
        print(f"   Available fields: {list(message.keys())}")
        
        has_reasoning_content = "reasoning_content" in message
        print(f"   Has reasoning_content field: {has_reasoning_content}")
        
        if has_reasoning_content:
            reasoning = message.get("reasoning_content")
            if reasoning:
                print(f"   reasoning_content length: {len(reasoning)} chars")
                print(f"   reasoning_content preview: {repr(reasoning[:100])}")
            else:
                print(f"   reasoning_content is empty/null")
        
        content_length = len(message["content"])
        print(f"   content length: {content_length} chars")
        
        print(f"   ✅ Response structure check complete")
        
    except Exception as e:
        print(f"   ❌ Response structure check failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mistral_compatibility())