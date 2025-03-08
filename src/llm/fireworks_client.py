import os
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any

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
    ) -> Dict[str, Any]:
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
            
        Returns:
            The API response as a dictionary
        """
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "temperature": temperature,
            "messages": messages
        }
        
        try:
            response = requests.post(
                self.BASE_URL, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Get a complete response from the model for a specific prompt.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters to pass to generate_completion
            
        Returns:
            The model's response content as a string
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.generate_completion(messages, **kwargs)
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to extract content from response: {str(e)}")