import os
import json
import requests
from typing import Dict, Any, Iterator, List, Optional, Union

from src.llm.base_client import ModelClient

class OpenAIModelClient(ModelClient):
    """
    Client for making API calls to OpenAI models.
    """
    
    BASE_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the OpenAI API client.
        
        Args:
            model_name: The name of the model to use (e.g., "gpt-4o")
            api_key: API key for OpenAI (if None, will use environment variable)
        """
        self.model_name = model_name
        
        # Get API key from args or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No OpenAI API key provided and OPENAI_API_KEY not found in environment")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_completion(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a completion from the OpenAI model.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }
        
        try:
            response = requests.post(
                self.BASE_URL, 
                headers=self.headers, 
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._process_stream(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenAI API request failed: {str(e)}")
    
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
                    if line.strip():  # Only log if line isn't empty
                        print(f"Error decoding JSON: {e}, Line: {line}")
    
    def generate_response(
        self, 
        prompt: str, 
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """Get a response from the model for a specific prompt."""
        messages = [{"role": "user", "content": prompt}]
        
        if stream:
            response_stream = self.generate_completion(messages, stream=True, **kwargs)
            return self._extract_streaming_content(response_stream)
        else:
            response = self.generate_completion(messages, stream=False, **kwargs)
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise Exception(f"Failed to extract content from OpenAI response: {str(e)}")
    
    def _extract_streaming_content(self, response_stream: Iterator[Dict[str, Any]]) -> Iterator[str]:
        """Extract content from a streaming response."""
        for chunk in response_stream:
            try:
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
                if content:
                    yield content
            except (KeyError, IndexError) as e:
                print(f"Error extracting content from chunk: {e}")