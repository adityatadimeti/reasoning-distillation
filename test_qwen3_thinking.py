#!/usr/bin/env python3
"""
Test script to verify Qwen3 thinking mode control on GPU cluster
"""
import requests
import json

def test_thinking_control():
    """Test thinking control on both reasoning and summarization servers"""
    
    print("=" * 60)
    print("TESTING QWEN3 THINKING MODE CONTROL")
    print("=" * 60)
    
    # Test prompt
    test_prompt = "What is 15 + 27? Show your work step by step."
    
    # Test reasoning server (should enable thinking)
    print("\n1. Testing REASONING server (port 8004) - thinking ENABLED:")
    print("-" * 50)
    
    try:
        response = requests.post("http://localhost:8004/v1/chat/completions", 
            json={
                "model": "Qwen/Qwen3-14B",
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 300,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": True}
            }, timeout=30)
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            print(f"Contains <think> tags: {'<think>' in content}")
            print(f"Contains </think> tags: {'</think>' in content}")
        else:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error connecting to reasoning server: {e}")
    
    # Test summarization server (should disable thinking)
    print("\n2. Testing SUMMARIZATION server (port 8005) - thinking DISABLED:")
    print("-" * 50)
    
    try:
        response = requests.post("http://localhost:8005/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3-14B", 
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 300,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}
            }, timeout=30)
            
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            print(f"Contains <think> tags: {'<think>' in content}")
            print(f"Contains </think> tags: {'</think>' in content}")
        else:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error connecting to summarization server: {e}")
    
    # Test without chat_template_kwargs (baseline)
    print("\n3. Testing WITHOUT chat_template_kwargs (baseline):")
    print("-" * 50)
    
    try:
        response = requests.post("http://localhost:8004/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3-14B", 
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 300,
                "temperature": 0
            }, timeout=30)
            
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            print(f"Contains <think> tags: {'<think>' in content}")
            print(f"Contains </think> tags: {'</think>' in content}")
        else:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("EXPECTED BEHAVIOR:")
    print("- Reasoning server: Should show <think></think> tags")
    print("- Summarization server: Should NOT show <think></think> tags")
    print("- Baseline: Depends on model default (likely shows thinking)")
    print("=" * 60)

if __name__ == "__main__":
    test_thinking_control()