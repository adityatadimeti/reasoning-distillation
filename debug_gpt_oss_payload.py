#!/usr/bin/env python3
"""
Debug script to see exactly what payload is being sent to vLLM for GPT-OSS
Similar to debug_client_payload.py but for GPT-OSS reasoning effort
"""
import asyncio
import sys
import os
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

async def debug_gpt_oss_payload():
    """Debug the exact payload being sent for GPT-OSS models"""
    
    print("=" * 60)
    print("DEBUGGING GPT-OSS CLIENT PAYLOAD")
    print("=" * 60)
    
    # Create GPT-OSS client
    client = VLLMModelClient(
        model_name="openai/gpt-oss-20b",
        host="localhost",
        port=8012,
        max_model_len=32768
    )
    
    print(f"Model name: {client.model_name}")
    print(f"Is GPT-OSS: {client.is_gpt_oss}")
    print(f"_is_gpt_oss_model(): {client._is_gpt_oss_model()}")
    
    # Test the payload generation by monkey-patching the request
    import aiohttp
    
    class DebugGPTOSSClient(VLLMModelClient):
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
            if self.is_qwen3 and qwen3_context:
                enable_thinking = qwen3_context == "reasoning"
                print(f"Qwen3 enable_thinking: {enable_thinking}")
                payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
            
            # Add GPT-OSS reasoning effort control via chat_template_kwargs
            gpt_oss_context = kwargs.get("gpt_oss_context")
            print(f"gpt_oss_context parameter: {gpt_oss_context}")
            print(f"self.is_gpt_oss: {self.is_gpt_oss}")
            
            if self.is_gpt_oss and gpt_oss_context:
                reasoning_effort = "high" if gpt_oss_context == "reasoning" else "low"
                print(f"reasoning_effort: {reasoning_effort}")
                payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
                print(f"Added chat_template_kwargs: {payload.get('chat_template_kwargs')}")
            else:
                print("No GPT-OSS chat_template_kwargs added")
                if not self.is_gpt_oss:
                    print("  - Reason: not a GPT-OSS model")
                if not gpt_oss_context:
                    print("  - Reason: no gpt_oss_context parameter")
            
            print(f"\nFull payload:")
            print(json.dumps(payload, indent=2))
            
            # Actually send the request to see the real response
            print(f"\nActual API response:")
            return await super().generate_completion_async(messages, max_tokens, temperature, stream, **kwargs)
    
    debug_client = DebugGPTOSSClient(
        model_name="openai/gpt-oss-20b",
        host="localhost",
        port=8012,
        max_model_len=32768
    )
    
    test_prompt = "What is 2+2?"
    
    # Test with reasoning context (high effort)
    print("\nTesting with gpt_oss_context='reasoning':")
    try:
        content, finish_reason, usage, cost, api_calls = await debug_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.6,
            gpt_oss_context="reasoning"
        )
        
        print(f"\nResponse: {content[:200]}...")
        print(f"Response length: {len(content)} characters")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with summarization context (low effort)
    print("\n" + "="*60)
    print("Testing with gpt_oss_context='summarization':")
    try:
        content, finish_reason, usage, cost, api_calls = await debug_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.6,
            gpt_oss_context="summarization"
        )
        
        print(f"\nResponse: {content[:200]}...")
        print(f"Response length: {len(content)} characters")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test without context (baseline)
    print("\n" + "="*60)
    print("Testing without gpt_oss_context (baseline):")
    try:
        content, finish_reason, usage, cost, api_calls = await debug_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.6
            # No gpt_oss_context parameter
        )
        
        print(f"\nResponse: {content[:200]}...")
        print(f"Response length: {len(content)} characters")
        
    except Exception as e:
        print(f"Error: {e}")

async def test_different_ports():
    """Test both servers to ensure they're configured correctly"""
    
    print("\n" + "=" * 60)
    print("TESTING BOTH SERVER PORTS")
    print("=" * 60)
    
    ports = [8012, 8013]
    contexts = ["reasoning", "summarization"]
    
    for port, context in zip(ports, contexts):
        print(f"\n--- Testing server on port {port} ---")
        
        debug_client = VLLMModelClient(
            model_name="openai/gpt-oss-20b",
            host="localhost",
            port=port,
            max_model_len=32768
        )
        
        try:
            # Test with minimal payload logging
            print(f"Testing {context} context on port {port}")
            
            content, finish_reason, usage, cost, api_calls = await debug_client.generate_response_async(
                prompt="What is 5+5?",
                max_tokens=100,
                temperature=0.1,
                gpt_oss_context=context
            )
            
            print(f"Response length: {len(content)} characters")
            print(f"Response preview: {content[:100]}...")
            
        except Exception as e:
            print(f"Error on port {port}: {e}")

if __name__ == "__main__":
    asyncio.run(debug_gpt_oss_payload())
    asyncio.run(test_different_ports())