#!/usr/bin/env python3
"""
Debug script to see exactly what payload is being sent to vLLM
"""
import asyncio
import sys
import os
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

async def debug_client_payload():
    """Debug the exact payload being sent"""
    
    print("=" * 60)
    print("DEBUGGING CLIENT PAYLOAD")
    print("=" * 60)
    
    # Create summarization client
    client = VLLMModelClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost",
        port=8005,
        max_model_len=32768
    )
    
    print(f"Model name: {client.model_name}")
    print(f"Is Qwen3: {client.is_qwen3}")
    print(f"_is_qwen3_model(): {client._is_qwen3_model()}")
    
    # Test the payload generation by monkey-patching the request
    import aiohttp
    
    class DebugVLLMClient(VLLMModelClient):
        async def generate_completion_async(self, messages, max_tokens, temperature, stream=False, **kwargs):
            """Override to show payload without sending request"""
            
            print("\n" + "-" * 40)
            print("PAYLOAD DEBUG")
            print("-" * 40)
            
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
            print(f"qwen3_context parameter: {qwen3_context}")
            print(f"self.is_qwen3: {self.is_qwen3}")
            
            if self.is_qwen3 and qwen3_context:
                enable_thinking = qwen3_context == "reasoning"
                print(f"enable_thinking: {enable_thinking}")
                payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
                print(f"Added chat_template_kwargs: {payload.get('chat_template_kwargs')}")
            else:
                print("No chat_template_kwargs added")
                if not self.is_qwen3:
                    print("  - Reason: not a Qwen3 model")
                if not qwen3_context:
                    print("  - Reason: no qwen3_context parameter")
            
            print(f"\nFull payload:")
            print(json.dumps(payload, indent=2))
            
            # Actually send the request to see the real response
            print(f"\nActual API response:")
            return await super().generate_completion_async(messages, max_tokens, temperature, stream, **kwargs)
    
    debug_client = DebugVLLMClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost",
        port=8005,
        max_model_len=32768
    )
    
    test_prompt = "What is 2+2?"
    
    # Test with summarization context
    print("\nTesting with qwen3_context='summarization':")
    try:
        content, finish_reason, usage, cost, api_calls = await debug_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.6,
            qwen3_context="summarization"
        )
        
        print(f"\nResponse: {content[:200]}...")
        print(f"Contains <think> tags: {'<think>' in content}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(debug_client_payload())