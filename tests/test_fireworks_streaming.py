import os
import sys
import pytest
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.fireworks_client import FireworksClient

# Load environment variables
load_dotenv()

def test_fireworks_client_initialization():
    """Test that the client initializes correctly."""
    client = FireworksClient()
    assert client.model_name == "accounts/fireworks/models/qwq-32b"
    assert client.api_key is not None

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_streaming_completion():
    """Test streaming completion to verify streaming API functionality."""
    client = FireworksClient()
    
    # Collect streamed chunks
    chunks = []
    full_response = ""
    
    # Get streaming response
    stream = client.generate_response(
        "Count from 1 to 5 slowly.",
        stream=True,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Collect and print chunks as they arrive
    for chunk in stream:
        chunks.append(chunk)
        full_response += chunk
        print(f"Received chunk: {chunk}", end="", flush=True)
    
    # Verify we got multiple chunks
    assert len(chunks) > 1, "Streaming should return multiple chunks"
    
    # Verify the combined response is meaningful
    assert len(full_response) > 0, "Combined response should not be empty"
    print(f"\nFull response: {full_response}")
    
    # Verify the response contains counting (basic content check)
    assert any(str(i) in full_response for i in range(1, 6)), "Response should contain numbers 1-5"

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_streaming_vs_non_streaming():
    """Compare streaming and non-streaming responses to ensure they're similar."""
    client = FireworksClient()
    prompt = "What is the capital of France?"
    
    # Get non-streaming response
    non_streaming_response = client.generate_response(
        prompt,
        max_tokens=1024,
        temperature=0.0,  # Use 0 temperature for deterministic output
        top_p=1.0,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Get streaming response
    streaming_chunks = []
    stream = client.generate_response(
        prompt,
        stream=True,
        max_tokens=1024,
        temperature=0.0,  # Use 0 temperature for deterministic output
        top_p=1.0,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Collect streaming response
    streaming_response = ""
    for chunk in stream:
        streaming_chunks.append(chunk)
        streaming_response += chunk
    
    # Print both responses for comparison
    print(f"Non-streaming response: {non_streaming_response}")
    print(f"Streaming response: {streaming_response}")
    
    # Verify both responses mention "Paris"
    assert "Paris" in non_streaming_response, "Non-streaming response should mention Paris"
    assert "Paris" in streaming_response, "Streaming response should mention Paris"
    
    # Verify we got multiple chunks
    assert len(streaming_chunks) > 1, "Streaming should return multiple chunks"

if __name__ == "__main__":
    # If run directly, execute the API tests
    if os.getenv("ENABLE_API_CALLS") == "1":
        print("Testing streaming completion...")
        test_streaming_completion()
        print("\nTesting streaming vs non-streaming...")
        test_streaming_vs_non_streaming()
    else:
        print("API calls disabled. Set ENABLE_API_CALLS=1 to run.")
