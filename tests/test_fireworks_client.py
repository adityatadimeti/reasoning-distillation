import os
import sys
import pytest
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.fireworks_client import FireworksModelClient

# Load environment variables
load_dotenv()

def test_fireworks_client_initialization():
    """Test that the client initializes correctly."""
    client = FireworksModelClient(model_name="accounts/fireworks/models/qwq-32b")
    assert client.model_name == "accounts/fireworks/models/qwq-32b"
    assert client.api_key is not None

@pytest.mark.skipif(os.getenv("ENABLE_API_CALLS") != "1", 
                   reason="API calls disabled")
def test_simple_completion():
    """Test a simple completion to verify API connectivity."""
    client = FireworksModelClient(model_name="accounts/fireworks/models/qwq-32b")
    response = client.generate_response(
        "What is 2+2?",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # Check that we got a non-empty response
    assert response is not None
    assert len(response) > 0
    print(f"Response: {response}")

if __name__ == "__main__":
    # If run directly, execute the API test
    if os.getenv("ENABLE_API_CALLS") == "1":
        test_simple_completion()
    else:
        print("API calls disabled. Set ENABLE_API_CALLS=1 to run.")