import os
import sys
import pytest
import json
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.fireworks_client import FireworksModelClient

# Load environment variables
load_dotenv()

def get_final_chunk_data(model, prompt, max_tokens):
    """
    Helper function to collect the last chunk from a streaming response.
    
    Args:
        model: The FireworksModelClient instance
        prompt: The prompt to send
        max_tokens: Maximum tokens for the response
        
    Returns:
        The last chunk from the streaming response
    """
    # Store all chunks to find the last one
    chunks = []
    
    # Generate streaming response
    stream = model.generate_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stream=True
    )
    
    # Collect all chunks
    for chunk in stream:
        chunks.append(chunk)
        
    # Return the last chunk if there are any
    if chunks:
        return chunks[-1]
    return None

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_streaming_finish_reason_length():
    """Test if the last chunk in streaming mode contains finish_reason='length'."""
    client = FireworksModelClient(model_name="accounts/fireworks/models/deepseek-r1")
    
    # Set up test parameters
    prompt = "Write a 500-word essay about artificial intelligence."
    max_tokens = 10  # Very small to trigger 'length'
    
    # Get the final chunk from the stream
    last_chunk = get_final_chunk_data(client, prompt, max_tokens)
    print(f"Last chunk from streaming with max_tokens={max_tokens}:")
    print(json.dumps(last_chunk, indent=2))
    
    # Get the complete response in non-streaming mode for comparison
    response = client.generate_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stream=False
    )
    print(f"Non-streaming response with max_tokens={max_tokens}:")
    print(json.dumps(response, indent=2))
    
    # Check if the finish_reason is present in the last chunk
    if last_chunk and 'choices' in last_chunk and last_chunk['choices']:
        has_finish_reason = 'finish_reason' in last_chunk['choices'][0]
        finish_reason_value = last_chunk['choices'][0].get('finish_reason')
        print(f"Last chunk has finish_reason: {has_finish_reason}")
        print(f"Last chunk finish_reason value: {finish_reason_value}")
    else:
        print("Last chunk doesn't have the expected structure")
    
    # Log the non-streaming finish_reason
    if 'choices' in response and response['choices']:
        print(f"Non-streaming finish_reason: {response['choices'][0].get('finish_reason')}")
    
@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_streaming_finish_reason_stop():
    """Test if the last chunk in streaming mode contains finish_reason='stop'."""
    client = FireworksModelClient(model_name="accounts/fireworks/models/deepseek-r1")
    
    # Set up test parameters
    prompt = "What is 2+2?"
    max_tokens = 100  # Enough for a full response
    
    # Get the final chunk from the stream
    last_chunk = get_final_chunk_data(client, prompt, max_tokens)
    print(f"Last chunk from streaming with max_tokens={max_tokens}:")
    print(json.dumps(last_chunk, indent=2))
    
    # Get the complete response in non-streaming mode for comparison
    response = client.generate_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stream=False
    )
    print(f"Non-streaming response with max_tokens={max_tokens}:")
    print(json.dumps(response, indent=2))
    
    # Check if the finish_reason is present in the last chunk
    if last_chunk and 'choices' in last_chunk and last_chunk['choices']:
        has_finish_reason = 'finish_reason' in last_chunk['choices'][0]
        finish_reason_value = last_chunk['choices'][0].get('finish_reason')
        print(f"Last chunk has finish_reason: {has_finish_reason}")
        print(f"Last chunk finish_reason value: {finish_reason_value}")
    else:
        print("Last chunk doesn't have the expected structure")
    
    # Log the non-streaming finish_reason
    if 'choices' in response and response['choices']:
        print(f"Non-streaming finish_reason: {response['choices'][0].get('finish_reason')}")

if __name__ == "__main__":
    # If run directly, execute the API tests
    if os.getenv("ENABLE_API_CALLS") == "1":
        print("\nTesting 'length' finish_reason in streaming mode:")
        test_streaming_finish_reason_length()
        
        print("\nTesting 'stop' finish_reason in streaming mode:")
        test_streaming_finish_reason_stop()
    else:
        print("API calls disabled. Set ENABLE_API_CALLS=1 to run.") 