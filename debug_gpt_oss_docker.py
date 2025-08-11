#!/usr/bin/env python3
"""
Debug script for GPT-OSS Docker containers
Shows exact payload being sent and monitors container logs
"""
import asyncio
import sys
import os
import json
import requests
import subprocess

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.vllm_client import VLLMModelClient

class DebugGPTOSSDockerClient(VLLMModelClient):
    """Debug version that shows exact payloads for Docker containers"""
    
    async def generate_completion_async(self, messages, max_tokens, temperature, stream=False, **kwargs):
        """Override to show payload before sending request"""
        
        print("\n" + "-" * 50)
        print("PAYLOAD DEBUG - DOCKER CONTAINER")
        print("-" * 50)
        
        # Show the parameters we received
        gpt_oss_context = kwargs.get("gpt_oss_context")
        print(f"gpt_oss_context parameter: {gpt_oss_context}")
        print(f"self.is_gpt_oss: {self.is_gpt_oss}")
        print(f"Port: {self.port}")
        
        # Build payload exactly like the real method
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
        
        # Add GPT-OSS reasoning effort control via chat_template_kwargs
        if self.is_gpt_oss and gpt_oss_context:
            reasoning_effort = "high" if gpt_oss_context == "reasoning" else "low"
            print(f"Adding reasoning_effort: {reasoning_effort}")
            payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
        else:
            print("No chat_template_kwargs added")
            if not self.is_gpt_oss:
                print("  - Reason: not a GPT-OSS model")
            if not gpt_oss_context:
                print("  - Reason: no gpt_oss_context parameter")
        
        print(f"\nFull payload being sent to {self.base_url}:")
        print(json.dumps(payload, indent=2))
        
        # Actually send the request
        print(f"\nSending request to Docker container...")
        return await super().generate_completion_async(messages, max_tokens, temperature, stream, **kwargs)

def check_docker_containers():
    """Check if Docker containers are running"""
    print("=" * 60)
    print("CHECKING DOCKER CONTAINERS")
    print("=" * 60)
    
    try:
        result = subprocess.run(["docker", "ps", "--filter", "name=gpt-oss"], 
                              capture_output=True, text=True)
        print("Running GPT-OSS containers:")
        print(result.stdout)
        
        if "gpt-oss-reasoning" not in result.stdout or "gpt-oss-summarization" not in result.stdout:
            print("\nWARNING: Not all containers are running!")
            print("Run: bash run_gpt_oss_20b_docker.sh")
            return False
            
    except Exception as e:
        print(f"Error checking containers: {e}")
        return False
    
    return True

def show_container_logs():
    """Show recent logs from both containers"""
    print("\n" + "=" * 60)
    print("CONTAINER LOGS")
    print("=" * 60)
    
    containers = ["gpt-oss-reasoning", "gpt-oss-summarization"]
    
    for container in containers:
        print(f"\n--- {container.upper()} LOGS ---")
        try:
            result = subprocess.run(["docker", "logs", "--tail", "10", container], 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
        except Exception as e:
            print(f"Error getting logs for {container}: {e}")

async def debug_payload_generation():
    """Debug the exact payloads being generated"""
    
    print("\n" + "=" * 60)
    print("DEBUGGING PAYLOAD GENERATION")
    print("=" * 60)
    
    test_prompt = "What is 8 + 12?"
    
    # Create debug clients
    reasoning_client = DebugGPTOSSDockerClient(
        model_name="openai/gpt-oss-20b",
        host="localhost",
        port=8012,
        max_model_len=32768
    )
    
    summarization_client = DebugGPTOSSDockerClient(
        model_name="openai/gpt-oss-20b",
        host="localhost",
        port=8013,
        max_model_len=32768
    )
    
    # Test reasoning context
    print("\n1. TESTING REASONING CONTEXT (high effort):")
    try:
        content, finish_reason, usage, cost, api_calls = await reasoning_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.1,
            gpt_oss_context="reasoning"
        )
        
        print(f"Response: {content}")
        print(f"Response length: {len(content)} characters")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test summarization context
    print("\n2. TESTING SUMMARIZATION CONTEXT (low effort):")
    try:
        content, finish_reason, usage, cost, api_calls = await summarization_client.generate_response_async(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.1,
            gpt_oss_context="summarization"
        )
        
        print(f"Response: {content}")
        print(f"Response length: {len(content)} characters")
        
    except Exception as e:
        print(f"Error: {e}")

def test_manual_api_calls():
    """Test manual API calls to verify Docker container responses"""
    
    print("\n" + "=" * 60)
    print("MANUAL API CALLS TO DOCKER CONTAINERS")
    print("=" * 60)
    
    test_cases = [
        {"port": 8012, "effort": "high", "name": "REASONING"},
        {"port": 8013, "effort": "low", "name": "SUMMARIZATION"},
    ]
    
    test_prompt = "Calculate 15 × 7 step by step."
    
    for case in test_cases:
        print(f"\n{case['name']} CONTAINER (port {case['port']}) - {case['effort']} effort:")
        print("-" * 50)
        
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": test_prompt}],
            "max_tokens": 250,
            "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": case['effort']}
        }
        
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(f"http://localhost:{case['port']}/v1/chat/completions",
                json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print(f"Status: {response.status_code}")
                print(f"Response: {content}")
                print(f"Response length: {len(content)} characters")
                
                # Check for reasoning indicators
                reasoning_words = ['first', 'then', 'step', 'multiply', 'calculate']
                reasoning_count = sum(1 for word in reasoning_words if word in content.lower())
                print(f"Reasoning indicators: {reasoning_count}")
                
            else:
                print(f"Error: HTTP {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Run all debug tests"""
    print("GPT-OSS Docker Debug Script")
    print("This will test Docker containers and show exact payloads")
    
    # Check if containers are running
    if not check_docker_containers():
        return
    
    # Show container logs
    show_container_logs()
    
    # Test manual API calls
    test_manual_api_calls()
    
    # Debug payload generation
    asyncio.run(debug_payload_generation())
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("Check the logs above to verify:")
    print("1. Both containers are running")
    print("2. chat_template_kwargs are being sent correctly")
    print("3. High vs low effort responses show differences")
    print("=" * 60)

if __name__ == "__main__":
    main()