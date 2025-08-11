#!/usr/bin/env python3
"""
Test script to verify GPT-OSS-20B reasoning effort control on GPU cluster
Direct API testing similar to the Qwen3 curl tests
"""
import requests
import json

def test_gpt_oss_reasoning_effort():
    """Test reasoning effort control on both reasoning and summarization servers"""
    
    print("=" * 60)
    print("TESTING GPT-OSS-20B REASONING EFFORT CONTROL")
    print("=" * 60)
    
    # Test prompt
    test_prompt = "What is 15 + 27? Show your work step by step."
    
    # Test reasoning server (should use high effort)
    print("\n1. Testing REASONING server (port 8012) - HIGH EFFORT:")
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
        print(f"Error connecting to reasoning server: {e}")
    
    # Test summarization server (should use low effort)
    print("\n2. Testing SUMMARIZATION server (port 8013) - LOW EFFORT:")
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
        print(f"Error connecting to summarization server: {e}")
    
    # Test without chat_template_kwargs (baseline)
    print("\n3. Testing WITHOUT chat_template_kwargs (baseline):")
    print("-" * 50)
    
    try:
        response = requests.post("http://localhost:8012/v1/chat/completions",
            json={
                "model": "openai/gpt-oss-20b", 
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 300,
                "temperature": 0
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
    
    # Test medium effort
    print("\n4. Testing MEDIUM effort (default):")
    print("-" * 50)
    
    try:
        response = requests.post("http://localhost:8012/v1/chat/completions",
            json={
                "model": "openai/gpt-oss-20b", 
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 300,
                "temperature": 0,
                "chat_template_kwargs": {"reasoning_effort": "medium"}
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
    
    print("\n" + "=" * 60)
    print("EXPECTED BEHAVIOR:")
    print("- High effort: Longer, more detailed responses with step-by-step reasoning")
    print("- Low effort: Shorter, more direct responses with minimal reasoning")
    print("- Medium effort: Balanced approach between high and low")
    print("- Baseline: Should default to medium effort behavior")
    print("=" * 60)

def test_complex_reasoning_task():
    """Test with a more complex reasoning task to better see effort differences"""
    
    print("\n" + "=" * 60)
    print("TESTING COMPLEX REASONING TASK")
    print("=" * 60)
    
    complex_prompt = """
    A farmer has chickens and rabbits. In total, there are 35 heads and 94 legs. 
    How many chickens and how many rabbits does the farmer have? 
    Show your complete reasoning process.
    """
    
    efforts = [("high", 8012), ("low", 8013)]
    
    for effort, port in efforts:
        print(f"\n{effort.upper()} EFFORT (port {port}):")
        print("-" * 40)
        
        try:
            response = requests.post(f"http://localhost:{port}/v1/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": complex_prompt}],
                    "max_tokens": 500,
                    "temperature": 0,
                    "chat_template_kwargs": {"reasoning_effort": effort}
                }, timeout=45)
                
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                print(f"Response length: {len(content)} characters")
                print(f"Response: {content[:200]}...")
                # Check for reasoning indicators
                reasoning_words = ['let', 'first', 'then', 'next', 'because', 'since', 'therefore', 'solve', 'equation']
                reasoning_count = sum(1 for word in reasoning_words if word in content.lower())
                print(f"Reasoning indicators found: {reasoning_count}")
            else:
                print(f"Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_gpt_oss_reasoning_effort()
    test_complex_reasoning_task()