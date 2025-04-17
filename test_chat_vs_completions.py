"""
Test script to compare outputs between:
1. Chat completions API
2. Completions API with apply_chat_template

This helps verify our implementation produces similar results as the original.
"""

import os
import json
import asyncio
import argparse
from dotenv import load_dotenv
import aiohttp
from transformers import AutoTokenizer
import difflib

# Import our tokenization utilities
from src.llm.tokenization import format_chat_for_completions, get_hf_model_name

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("FIREWORKS_API_KEY")

async def call_chat_completions_api(model_name, messages, temperature=0.0, max_tokens=1000):
    """Call the Fireworks Chat Completions API directly."""
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    print(f"Calling Chat Completions API with {len(messages)} messages...")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            return await response.json()

async def call_completions_api(model_name, prompt, temperature=0.0, max_tokens=1000):
    """Call the Fireworks Completions API directly."""
    url = "https://api.fireworks.ai/inference/v1/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    print(f"Calling Completions API with prompt (length: {len(prompt)} chars)...")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            return await response.json()

def compare_outputs(chat_output, completions_output):
    """Compare the outputs from both API calls and highlight differences."""
    # Extract the actual content
    chat_content = chat_output["choices"][0]["message"]["content"]
    completions_content = completions_output["choices"][0]["text"]
    
    # Print both outputs
    print("\nChat Completions API output:")
    print("-" * 80)
    print(chat_content)
    print("-" * 80)
    
    print("\nCompletions API with chat template output:")
    print("-" * 80)
    print(completions_content)
    print("-" * 80)
    
    # Show a diff of the two outputs
    diff = difflib.unified_diff(
        chat_content.splitlines(),
        completions_content.splitlines(),
        lineterm='',
        n=3  # Context lines
    )
    
    print("\nDifferences (if any):")
    print("-" * 80)
    diff_lines = list(diff)
    if diff_lines:
        for line in diff_lines:
            print(line)
    else:
        print("No differences found! The outputs are identical.")
    print("-" * 80)
    
    return chat_content, completions_content

async def main():
    parser = argparse.ArgumentParser(description="Test Chat vs Completions API for equivalence")
    parser.add_argument("--model", default="accounts/vivek-vajipey-84a360/deployedModels/deepseek-r1-distill-qwen-14b-61e7dbf1", 
                       help="Fireworks model name")
    args = parser.parse_args()
    
    # Define a sample conversation to test with
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves math problems step-by-step."},
        {"role": "user", "content": "What is 17 + 25?"}
    ]
    
    # Get the corresponding HF model name for the template
    model_name = args.model
    hf_model_name = get_hf_model_name(model_name)
    print(f"Using model: {model_name}")
    print(f"HuggingFace model name: {hf_model_name}")
    
    # Format messages for completions API using our formatter
    formatted_prompt = format_chat_for_completions(messages, model_name)
    
    # Print the formatted prompt for inspection
    print("\nFormatted prompt for completions API:")
    print("-" * 80)
    print(formatted_prompt)
    print("-" * 80)
    
    # Make both API calls
    chat_output = await call_chat_completions_api(model_name, messages)
    completions_output = await call_completions_api(model_name, formatted_prompt)
    
    # Compare the outputs
    chat_content, completions_content = compare_outputs(chat_output, completions_output)
    
    # Save the outputs for reference
    os.makedirs("api_comparison_results", exist_ok=True)
    with open("api_comparison_results/chat_output.json", "w") as f:
        json.dump(chat_output, f, indent=2)
    with open("api_comparison_results/completions_output.json", "w") as f:
        json.dump(completions_output, f, indent=2)
    with open("api_comparison_results/chat_content.txt", "w") as f:
        f.write(chat_content)
    with open("api_comparison_results/completions_content.txt", "w") as f:
        f.write(completions_content)

if __name__ == "__main__":
    asyncio.run(main())
