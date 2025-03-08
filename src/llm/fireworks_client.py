import os
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Iterator, Optional, Union

class FireworksClient:
    """
    A lightweight client for making API calls to Fireworks AI models.
    """
    
    BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    def __init__(self, model_name: str = "accounts/fireworks/models/qwq-32b"):
        """
        Initialize the Fireworks API client.
        
        Args:
            model_name: The name of the model to use (default: QwQ 32B)
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment variables")
        
        self.model_name = model_name
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
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a completion from the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            presence_penalty: Penalty for token presence
            frequency_penalty: Penalty for token frequency
            stream: Whether to stream the response
            
        Returns:
            If stream=False: The complete API response as a dictionary
            If stream=True: An iterator yielding response chunks
        """
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "temperature": temperature,
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
            response.raise_for_status()  # Raise exception for HTTP errors
            
            if stream:
                return self._process_stream(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def _process_stream(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """
        Process a streaming response from the API.
        
        Args:
            response: The streaming response from requests
            
        Yields:
            Parsed JSON chunks from the stream
        """
        for line in response.iter_lines():
            if line:
                # Remove 'data: ' prefix and parse JSON
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]  # Remove 'data: ' prefix
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
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Get a complete response from the model for a specific prompt.
        
        Args:
            prompt: The prompt to send to the model
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to generate_completion
            
        Returns:
            If stream=False: The model's complete response content as a string
            If stream=True: An iterator yielding response content chunks
        """
        messages = [{"role": "user", "content": prompt}]
        
        if stream:
            response_stream = self.generate_completion(messages, stream=True, **kwargs)
            return self._extract_streaming_content(response_stream)
        else:
            response = self.generate_completion(messages, **kwargs)
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise Exception(f"Failed to extract content from response: {str(e)}")
    
    def _extract_streaming_content(self, response_stream: Iterator[Dict[str, Any]]) -> Iterator[str]:
        """
        Extract content from a streaming response.
        
        Args:
            response_stream: Iterator of response chunks
            
        Yields:
            Content chunks from the response
        """
        for chunk in response_stream:
            try:
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content
            except (KeyError, IndexError) as e:
                print(f"Error extracting content from chunk: {e}")