#!/usr/bin/env python3
"""
Debug script to check what payload is being sent to vLLM for Qwen3
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

# Monkey patch the vLLM client to log the exact payload being sent
original_generate_completion_async = VLLMModelClient.generate_completion_async

async def debug_generate_completion_async(self, messages, max_tokens, temperature, stream=False, **kwargs):
    """Debug version that logs the payload"""
    
    # Prepare the request payload (same as original)
    payload = {
        "model": self.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "top_p": kwargs.get("top_p", 1.0),
        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
        "presence_penalty": kwargs.get("presence_penalty", 0.0),
    }
    
    # Add Qwen3 thinking mode control via chat_template_kwargs
    qwen3_context = kwargs.get("qwen3_context")
    print(f"\n=== QWEN3 DEBUG INFO ===")
    print(f"Model name: {self.model_name}")
    print(f"is_qwen3: {self.is_qwen3}")
    print(f"qwen3_context parameter: {qwen3_context}")
    
    if self.is_qwen3 and qwen3_context:
        enable_thinking = qwen3_context == "reasoning"
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        print(f"Added chat_template_kwargs: {payload['chat_template_kwargs']}")
    else:
        print("No chat_template_kwargs added")
        if not self.is_qwen3:
            print("  - Reason: not a Qwen3 model")
        if not qwen3_context:
            print("  - Reason: no qwen3_context parameter")
    
    # Add optional parameters if provided
    if "top_k" in kwargs and kwargs["top_k"] is not None:
        payload["top_k"] = kwargs["top_k"]
    if "stop" in kwargs:
        payload["stop"] = kwargs["stop"]
    
    print(f"\n=== FULL PAYLOAD ===")
    print(json.dumps(payload, indent=2))
    print("=" * 50)
    
    # Call the original method
    return await original_generate_completion_async(self, messages, max_tokens, temperature, stream, **kwargs)

# Apply the monkey patch
VLLMModelClient.generate_completion_async = debug_generate_completion_async

# Now test it
async def test_qwen3_payload():
    """Test what payload gets sent for Qwen3"""
    
    print("Testing Qwen3 payload generation...")
    
    # Create client
    client = VLLMModelClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost", 
        port=8005,
        max_model_len=32768
    )
    
    # Test simple prompt
    prompt = "Using all numbers from [95, 89, 47, 24], create an equation that equals 17."
    
    print("\n1. Testing with qwen3_context='reasoning':")
    try:
        # This should trigger the debug logging
        response = await client.generate_response_async(
            prompt=prompt,
            max_tokens=100,
            temperature=0.6,
            qwen3_context="reasoning"  # This should enable thinking
        )
        print(f"Response: {response[0][:100]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_qwen3_payload())