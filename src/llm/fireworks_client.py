import os
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Iterator, Optional, Union

from src.llm.base_client import ModelClient

class FireworksModelClient(ModelClient):
    """
    Client for making API calls to Fireworks AI models.
    """
    
    BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the Fireworks API client.
        
        Args:
            model_name: The name of the model to use
            api_key: API key for Fireworks (if None, will use environment variable)
        """
        # Load environment variables
        load_dotenv()
        
        self.model_name = model_name
        
        # Get API key from args or environment
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("No Fireworks API key provided and FIREWORKS_API_KEY not found in environment")
        
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_completion(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence_penalty: float,
        frequency_penalty: float,
        stream: bool = False,   
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a completion from the Fireworks model.
        """
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(
                self.BASE_URL, 
                headers=self.headers, 
                data=json.dumps(payload),
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._process_stream(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Fireworks API request failed: {str(e)}")
    
    def _process_stream(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """Process a streaming response from the API."""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]
                if line == '[DONE]':
                    break
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}, Line: {line}")
    
    def generate_response(
        self, 
        prompt: str, 
        stream: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """Get a response from the model for a specific prompt."""
        messages = [{"role": "user", "content": prompt}]
        
        if verbose:
            print(f"\n[VERBOSE] Fireworks API Request to {self.model_name}")
            print(f"[VERBOSE] Messages: {json.dumps(messages, indent=2)}")
            print(f"[VERBOSE] Parameters: {json.dumps({k: v for k, v in kwargs.items()}, indent=2)}")
        
        if stream:
            response_stream = self.generate_completion(messages, stream=True, **kwargs)
            return self._extract_streaming_content(response_stream)
        else:
            response = self.generate_completion(messages, stream=False, **kwargs)
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise Exception(f"Failed to extract content from Fireworks response: {str(e)}")
    
    def _extract_streaming_content(self, response_stream: Iterator[Dict[str, Any]]) -> Iterator[str]:
        """Extract content from a streaming response."""
        for chunk in response_stream:
            try:
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content
            except (KeyError, IndexError) as e:
                print(f"Error extracting content from chunk: {e}")
                # Don't let errors break the stream, yield an empty string
                yield ""