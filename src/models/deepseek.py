"""
Implementation for DeepSeek's R1 model.
"""
import os
import json
import time
import random
import requests
from typing import Dict, List, Optional, Any
import re

from src.utils.config import Config
from src.models.base import Model, ModelResponse

class DeepSeekModel(Model):
    """
    Model class for interacting with DeepSeek R1 via the Fireworks API.
    """
    def __init__(self, config: Config):
        """
        Initialize the DeepSeek model.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        # Load API key from environment or config
        self.api_key = os.environ.get("FIREWORKS_API_KEY") or self.model_config.get("api_key")
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment or config")
        
        # API settings
        self.api_url = self.model_config.get("api_url", "https://api.fireworks.ai/inference/v1/completions")
        self.model_id = self.model_config.get("model_id", "accounts/fireworks/models/deepseek-r1")
        
        # Headers for API requests
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Reasoning settings
        self.reasoning_config = config.get("reasoning", {})
        self.think_tag = self.reasoning_config.get("think_tag", True)
        self.max_extensions = self.reasoning_config.get("max_extensions", 10)
        self.target_token_count = self.reasoning_config.get("target_token_count", 2000)
        
        # Continuation phrases for extending reasoning
        self.continuation_phrases = [
            "Let me think more about this.",
            "Actually, I should reconsider my approach.",
            "Let me analyze this further.",
            "Let me double-check this calculation.",
            "I need to think about this from another angle.",
            "Let me explore an alternative solution method.",
            "I should verify these results with another approach.",
            "Let me ensure the reasoning so far is correct.",
            "Let me consider if there are any edge cases.",
            "I'll try a different way to solve this problem."
        ]
    
    def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make an API call to the Fireworks API with retry logic.
        
        Args:
            payload: API request payload
            
        Returns:
            API response as a dictionary
            
        Raises:
            Exception: If API call fails after all retries
        """
        # Check for API calls safeguard
        if os.environ.get("ENABLE_API_CALLS") != "1":
            raise EnvironmentError(
                "API calls are disabled. Set ENABLE_API_CALLS=1 to enable real API calls. "
                "This safeguard prevents accidental usage of paid API services during testing."
            )

        for attempt in range(self.retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt) * self.retry_delay
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API error: {response.status_code}")
                    print(response.text)
                    if attempt == self.retries - 1:
                        break
                    time.sleep(self.retry_delay)
            except Exception as e:
                print(f"Error: {str(e)}")
                if attempt == self.retries - 1:
                    raise
                time.sleep(self.retry_delay)
        
        raise Exception(f"API call failed after {self.retries} attempts")
    
    def generate(self, 
                prompt: str, 
                max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None,
                **kwargs) -> ModelResponse:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The input prompt to generate from
            max_tokens: Maximum number of tokens to generate (overrides config)
            temperature: Temperature for generation (overrides config)
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse containing the generated text and metadata
        """
        # Prepare API payload
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", 40),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty)
        }
        
        # Make API call
        response_data = self._make_api_call(payload)
        
        # Extract the generated text
        if "choices" in response_data and len(response_data["choices"]) > 0:
            generation = response_data["choices"][0]["text"]
        else:
            raise ValueError("No text generation found in response")
        
        # Get token usage if available
        tokens_used = None
        if "usage" in response_data:
            tokens_used = response_data["usage"].get("total_tokens")
        
        return ModelResponse(
            text=generation,
            prompt=prompt,
            tokens_used=tokens_used,
            model_name=self.model_id,
            raw_response=response_data
        )
    
    def generate_reasoning(self, 
                          question: str, 
                          max_extensions: Optional[int] = None,
                          target_token_count: Optional[int] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate extensive reasoning trace for a question using the think tag approach.
        
        Args:
            question: The input question to reason about
            max_extensions: Maximum number of reasoning extensions (overrides config)
            target_token_count: Target number of tokens for reasoning (overrides config)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing the reasoning trace, answer, and metadata
        """
        max_extensions = max_extensions or self.max_extensions
        target_token_count = target_token_count or self.target_token_count
        
        # Initialize with thinking prompt
        initial_prompt = f"{question}\n\n<think>" if self.think_tag else question
        
        full_reasoning = ""
        num_extensions = 0
        
        current_prompt = initial_prompt
        
        # Generate initial reasoning and continue extending if needed
        while num_extensions <= max_extensions:
            # Prepare and make API call
            payload = {
                "model": self.model_id,
                "max_tokens": 1000,  # Using a smaller chunk size per call
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "top_k": kwargs.get("top_k", 40),
                "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
                "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
                "prompt": current_prompt
            }
            
            response_data = self._make_api_call(payload)
            
            # Extract the generated text
            if "choices" in response_data and len(response_data["choices"]) > 0:
                generation = response_data["choices"][0]["text"]
            else:
                raise ValueError("No generation found in response")
            
            # Check if the model completed its thinking with </think> tag
            if self.think_tag and "</think>" in generation:
                # Remove everything from </think> onwards for continuation
                thinking_part = generation.split("</think>")[0]
                full_reasoning += thinking_part
                break
            else:
                # No closing tag, just append the generation
                full_reasoning += generation
            
            # Estimate the current token count
            current_token_count = self.estimate_tokens(full_reasoning)
            
            # Check if we've reached the target token count
            if current_token_count >= target_token_count:
                break
            
            # Haven't reached target count, continue extending
            num_extensions += 1
            if num_extensions <= max_extensions:
                # Add a continuation phrase
                continuation = random.choice(self.continuation_phrases)
                
                # Explicitly add the continuation phrase to the full reasoning
                full_reasoning += f"\n{continuation}\n"
                
                # Update the current prompt with the full reasoning including continuation
                if self.think_tag:
                    current_prompt = initial_prompt + full_reasoning
                else:
                    current_prompt = question + "\n\n" + full_reasoning
            else:
                # Reached max extensions, close the reasoning
                break
        
        # After reaching target token count or max extensions, generate a final answer
        if self.think_tag:
            final_prompt = f"{question}\n\n<think>{full_reasoning}</think>"
        else:
            final_prompt = f"{question}\n\n{full_reasoning}\n\nTherefore, the answer is:"
        
        final_payload = {
            "model": self.model_id,
            "max_tokens": 500,
            "temperature": kwargs.get("temperature", self.temperature),
            "prompt": final_prompt
        }
        
        final_response = self._make_api_call(final_payload)
        
        if "choices" in final_response and len(final_response["choices"]) > 0:
            final_answer = final_response["choices"][0]["text"].strip()
        else:
            final_answer = ""
        
        # Calculate the final token count before returning
        current_token_count = self.estimate_tokens(full_reasoning)
        
        # Construct and return the final output
        return {
            "question": question,
            "reasoning": full_reasoning,
            "full_reasoning_with_tags": f"<think>{full_reasoning}</think>" if self.think_tag else full_reasoning,
            "answer": final_answer,
            "extensions": num_extensions,
            "estimated_token_count": current_token_count
        }
    
    def summarize_reasoning(self, reasoning_trace: str, **kwargs) -> str:
        """
        Use DeepSeek R1 to summarize its own reasoning trace.
        
        Args:
            reasoning_trace: The reasoning trace to summarize
            **kwargs: Additional model-specific parameters
            
        Returns:
            Summarized reasoning trace
        """
        # Construct a prompt for summarization
        summarization_prompt = (
            "Summarize the following reasoning trace into a concise form that preserves "
            "all key steps and important calculations:\n\n"
            f"{reasoning_trace}\n\n"
            "Summary:"
        )
        
        # Generate summary
        response = self.generate(
            prompt=summarization_prompt,
            max_tokens=kwargs.get("max_tokens", min(1000, self.max_tokens // 2)),
            temperature=kwargs.get("temperature", 0.5)  # Lower temperature for more focused summary
        )
        
        return response.text.strip()

    def extract_answer(self, text: str, flexible_extraction: bool = False) -> str:
        """
        Extract the answer from model output, looking for boxed notation.
        
        Args:
            text: Text containing the answer
            flexible_extraction: If True, use more flexible patterns to extract answers
            
        Returns:
            Extracted answer or empty string if not found
        """
        # Look for \boxed{X} pattern
        pattern = r'\\boxed\{(.*?)\}'
        matches = re.findall(pattern, text)
        
        if matches:
            return matches[0].strip()
            
        # If flexible extraction is enabled, try additional patterns
        if flexible_extraction:
            # Look for "**Answer:** X" pattern (common in markdown formatting)
            bold_pattern = r'\*\*Answer:\*\*\s*(.*?)(?:\.|$)'
            bold_matches = re.findall(bold_pattern, text)
            if bold_matches:
                return bold_matches[0].strip()
                
            # Look for "Answer: X" pattern (without bold)
            simple_pattern = r'Answer:\s*(.*?)(?:\.|$)'
            simple_matches = re.findall(simple_pattern, text)
            if simple_matches:
                return simple_matches[0].strip()
                
            # Look for a number followed by units (e.g., "180 miles")
            units_pattern = r'(\d+)\s*(?:miles|km|meters|m|feet|ft)'
            units_matches = re.findall(units_pattern, text)
            if units_matches:
                return units_matches[0].strip()
                
            # Last resort: try to find any number in the text
            number_pattern = r'(\d+)'
            number_matches = re.findall(number_pattern, text)
            if number_matches:
                return number_matches[0].strip()
        
        return ""