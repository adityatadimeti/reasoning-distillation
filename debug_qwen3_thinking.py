#!/usr/bin/env python3
"""
Debug script to test Qwen3 thinking mode control via chat_template_kwargs
"""
import asyncio
import aiohttp
import json

async def test_qwen3_thinking_modes():
    """Test both thinking and non-thinking modes"""
    
    # Test with enable_thinking=True (should have <think> tags)
    print("=== Testing enable_thinking=True (Reasoning Mode) ===")
    payload_thinking = {
        "model": "Qwen/Qwen3-14B",
        "messages": [{"role": "user", "content": "Solve: What is 2+2?"}],
        "max_tokens": 200,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": True}
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8004/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload_thinking
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"Response: {content[:200]}...")
                print(f"Has <think> tags: {'<think>' in content}")
            else:
                print(f"Error: {response.status}")
                print(await response.text())
    
    print("\n" + "="*50 + "\n")
    
    # Test with enable_thinking=False (should NOT have <think> tags)  
    print("=== Testing enable_thinking=False (Summarization Mode) ===")
    payload_no_thinking = {
        "model": "Qwen/Qwen3-14B", 
        "messages": [{"role": "user", "content": "Solve: What is 2+2?"}],
        "max_tokens": 200,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8005/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload_no_thinking
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"Response: {content[:200]}...")
                print(f"Has <think> tags: {'<think>' in content}")
            else:
                print(f"Error: {response.status}")
                print(await response.text())

    # Also test what our vLLM client is actually sending
    print("\n" + "="*50 + "\n")
    print("=== Testing our vLLM Client Implementation ===")
    
    from src.llm.vllm_client import VLLMModelClient
    
    # Test reasoning client
    reasoning_client = VLLMModelClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost", 
        port=8004
    )
    
    print(f"Is Qwen3 model: {reasoning_client.is_qwen3}")
    
    # Test summarization client  
    summarization_client = VLLMModelClient(
        model_name="Qwen/Qwen3-14B",
        host="localhost",
        port=8005
    )
    
    try:
        # Test reasoning call
        reasoning_response = await reasoning_client.generate_response_async(
            "Solve: What is 2+2?",
            max_tokens=200,
            temperature=0.6,
            qwen3_context="reasoning"
        )
        print(f"Reasoning response has <think>: {'<think>' in reasoning_response[0]}")
        
        # Test summarization call
        summary_response = await summarization_client.generate_response_async(
            "Summarize: 2+2=4 because we add two and two together.", 
            max_tokens=200,
            temperature=0.6,
            qwen3_context="summarization"
        )
        print(f"Summary response has <think>: {'<think>' in summary_response[0]}")
        print(f"Summary content: {summary_response[0][:100]}...")
        
    except Exception as e:
        print(f"Error testing client: {e}")

if __name__ == "__main__":
    asyncio.run(test_qwen3_thinking_modes())