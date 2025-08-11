#!/usr/bin/env python3
"""
Test script to verify GPT-OSS-20B Docker containers and reasoning effort control
Tests both direct API calls and vLLM client integration
"""
import asyncio
import sys
import os
import requests
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

def test_docker_containers():
    """Test that both Docker containers are running and accessible"""
    
    print("=" * 60)
    print("TESTING GPT-OSS DOCKER CONTAINERS")
    print("=" * 60)
    
    ports = [8012, 8013]
    container_names = ["reasoning", "summarization"]
    
    for port, name in zip(ports, container_names):
        print(f"\nTesting {name} container on port {port}:")
        print("-" * 40)
        
        try:
            # Test health endpoint
            health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
            print(f"Health check: {health_response.status_code}")
            
            # Test models endpoint
            models_response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                model_names = [model["id"] for model in models_data["data"]]
                print(f"Available models: {model_names}")
            else:
                print(f"Models endpoint error: {models_response.status_code}")
                
        except Exception as e:
            print(f"Error connecting to {name} container: {e}")
            return False
    
    return True

def test_direct_api_calls():
    """Test direct API calls with chat_template_kwargs"""
    
    print("\n" + "=" * 60)
    print("TESTING DIRECT API CALLS WITH REASONING EFFORT")
    print("=" * 60)
    
    test_prompt = "What is 25 + 37? Show your work step by step."
    
    # Test high effort (reasoning server)
    print("\n1. DIRECT API - HIGH EFFORT (port 8012):")
    print("-" * 50)
    
    try:
        response = requests.post("http://localhost:8012/v1/chat/completions", 
            json={
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 300,
                "temperature": 0,
                "chat_template_kwargs": {"reasoning_effort": "high"}
            }, timeout=30)
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            print(f"Response length: {len(content)} characters")
            print(f"Contains detailed reasoning: {'step' in content.lower() or 'first' in content.lower()}")
        else:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test low effort (summarization server)
    print("\n2. DIRECT API - LOW EFFORT (port 8013):")
    print("-" * 50)
    
    try:
        response = requests.post("http://localhost:8013/v1/chat/completions",
            json={
                "model": "openai/gpt-oss-20b", 
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 300,
                "temperature": 0,
                "chat_template_kwargs": {"reasoning_effort": "low"}
            }, timeout=30)
            
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            print(f"Response length: {len(content)} characters")
            print(f"Contains detailed reasoning: {'step' in content.lower() or 'first' in content.lower()}")
        else:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")

async def test_vllm_client():
    """Test vLLM client integration with Docker containers"""
    
    print("\n" + "=" * 60)
    print("TESTING VLLM CLIENT WITH DOCKER CONTAINERS")
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
    
    print(f"Reasoning client - Is GPT-OSS: {reasoning_client.is_gpt_oss}")
    print(f"Summarization client - Is GPT-OSS: {summarization_client.is_gpt_oss}")
    
    test_prompt = "What is 42 + 58? Show your work step by step."
    
    # Test reasoning client (high effort)
    print("\n1. VLLM CLIENT - REASONING (gpt_oss_context='reasoning'):")
    print("-" * 50)
    
    try:
        content, finish_reason, usage, cost, api_calls = await reasoning_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=300,
            temperature=0.1,
            gpt_oss_context="reasoning"  # Should enable high effort
        )
        
        print(f"Response: {content}")
        print(f"Response length: {len(content)} characters")
        print(f"Contains detailed reasoning: {'step' in content.lower() or 'first' in content.lower()}")
        print(f"Finish reason: {finish_reason}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test summarization client (low effort)
    print("\n2. VLLM CLIENT - SUMMARIZATION (gpt_oss_context='summarization'):")
    print("-" * 50)
    
    try:
        content, finish_reason, usage, cost, api_calls = await summarization_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=300, 
            temperature=0.1,
            gpt_oss_context="summarization"  # Should enable low effort
        )
        
        print(f"Response: {content}")
        print(f"Response length: {len(content)} characters")
        print(f"Contains detailed reasoning: {'step' in content.lower() or 'first' in content.lower()}")
        print(f"Finish reason: {finish_reason}")
        
    except Exception as e:
        print(f"Error: {e}")

async def test_complex_reasoning():
    """Test with a complex problem to see effort differences"""
    
    print("\n" + "=" * 60)
    print("TESTING COMPLEX REASONING - HIGH VS LOW EFFORT")
    print("=" * 60)
    
    complex_prompt = """
    A store sells apples for $2 each and oranges for $3 each. 
    If someone buys 7 fruits total and spends $18, how many apples and oranges did they buy?
    Solve this step by step.
    """
    
    # Create clients
    reasoning_client = VLLMModelClient(model_name="openai/gpt-oss-20b", host="localhost", port=8012)
    summarization_client = VLLMModelClient(model_name="openai/gpt-oss-20b", host="localhost", port=8013)
    
    # Test high effort
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
        print(f"Response preview: {content[:200]}...")
        print(f"Full response: {content}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test low effort
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
        print(f"Response preview: {content[:200]}...")
        print(f"Full response: {content}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all Docker container tests"""
    print("Starting GPT-OSS Docker Container Tests...")
    
    # Test container connectivity first
    if not test_docker_containers():
        print("\nERROR: Docker containers are not accessible!")
        print("Please run: bash run_gpt_oss_20b_docker.sh")
        return
    
    # Test direct API calls
    test_direct_api_calls()
    
    # Test vLLM client integration
    asyncio.run(test_vllm_client())
    
    # Test complex reasoning comparison
    asyncio.run(test_complex_reasoning())
    
    print("\n" + "=" * 60)
    print("Docker tests completed!")
    print("If both high and low effort show different response patterns,")
    print("the GPT-OSS reasoning effort control is working correctly.")
    print("=" * 60)

if __name__ == "__main__":
    main()