"""
Implementation for models on the Fireworks AI platform.
"""
import os
import json
import time
import random
import requests
from typing import Dict, List, Optional, Any
import re
import logging

# Import OpenAI for API compatibility
import openai
from openai import OpenAI

from src.utils.config import Config
from src.models.base import Model, ModelResponse

# Import experiment monitor for tracking API messages
try:
    from src.utils.experiment_monitor import add_api_message
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

logger = logging.getLogger(__name__)

class FireworksModel(Model):
    """
    Model class for interacting with Fireworks AI models via their API.
    """
    def __init__(self, config: Config, model_config_key: str = "model"):
        """
        Initialize the Fireworks model.
        
        Args:
            config: Configuration object
            model_config_key: Key to access model-specific config
        """
        super().__init__(config, model_config_key)
        
        # Load API key from environment or config
        self.api_key = os.environ.get("FIREWORKS_API_KEY") or self.model_config.get("api_key")
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment or config")
        
        # API settings
        self.api_base = self.model_config.get("api_base", "https://api.fireworks.ai/inference/v1")
        self.completion_url = f"{self.api_base}/completions"
        self.chat_url = f"{self.api_base}/chat/completions"
        
        # Use model_id from config or construct from name
        if not self.model_id and self.model_name:
            # For convenience, allow specifying just the model name
            self.model_id = f"accounts/fireworks/models/{self.model_name}"
        elif not self.model_id:
            raise ValueError("Model ID not found in config")
        
        # Headers for API requests
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Initialize OpenAI client for compatibility
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
        # Reasoning settings
        self.reasoning_config = config.get("reasoning", {})
        self.think_tag = self.reasoning_config.get("think_tag", True)
        self.max_extensions = self.reasoning_config.get("max_extensions", 10)
        self.target_token_count = self.reasoning_config.get("target_token_count", 2000)
        
        # Continuation phrases for extending reasoning
        self.continuation_phrases = self.reasoning_config.get("continuation_phrases", [
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
        ])
        
        logger.info(f"Initialized FireworksModel with model ID: {self.model_id}")
    
    def _check_api_safeguard(self):
        """
        Check if API calls are enabled.
        
        Raises:
            EnvironmentError: If API calls are disabled
        """
        # Skip this check in production environment
        if os.environ.get("ENVIRONMENT") == "production":
            return
            
        if os.environ.get("ENABLE_API_CALLS") != "1":
            raise EnvironmentError(
                "API calls are disabled. Set ENABLE_API_CALLS=1 to enable real API calls. "
                "This safeguard prevents accidental usage of paid API services during testing."
            )
    
    def _make_api_call(self, url: str, payload: Dict[str, Any], problem_id: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Make an API call to the Fireworks API with retry logic.
        
        Args:
            url: API endpoint URL
            payload: API request payload
            problem_id: Optional problem ID for tracking
            iteration: Optional iteration number for tracking
            
        Returns:
            API response as a dictionary
            
        Raises:
            Exception: If API call fails after all retries
        """
        # Check API calls safeguard
        self._check_api_safeguard()
        
        # Track API request if monitor is available
        if MONITOR_AVAILABLE and problem_id is not None and iteration is not None:
            add_api_message(problem_id, iteration, "request", payload.get("messages", [payload]))
        
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    url, 
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Track API response if monitor is available
                    if MONITOR_AVAILABLE and problem_id is not None and iteration is not None:
                        add_api_message(problem_id, iteration, "response", [response_data])
                    
                    return response_data
                elif response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt) * self.retry_delay
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error: {response.status_code}")
                    logger.error(response.text)
                    if attempt == self.retries - 1:
                        break
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error: {str(e)}")
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
        response_data = self._make_api_call(self.completion_url, payload)
        
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
    
    def chat_completion(self,
                       messages: List[Dict[str, str]],
                       max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None,
                       problem_id: Optional[str] = None,
                       iteration: Optional[int] = None,
                       **kwargs) -> ModelResponse:
        """
        Generate a chat completion using the Fireworks API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            problem_id: Optional problem ID for tracking
            iteration: Optional iteration number for tracking
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            ModelResponse object with the generated text
        """
        # Use default values from config if not provided
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        # Prepare the payload
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        # Make the API call
        try:
            response = self._make_api_call(self.chat_url, payload, problem_id, iteration)
            
            # Extract the generated text
            if "choices" in response and len(response["choices"]) > 0:
                text = response["choices"][0]["message"]["content"]
                return ModelResponse(text=text, raw_response=response)
            else:
                logger.error(f"Unexpected response format: {response}")
                return ModelResponse(text="", raw_response=response)
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            return ModelResponse(text="", error=str(e))
    
    def generate_reasoning(self, 
                          question: str, 
                          max_extensions: Optional[int] = None,
                          target_token_count: Optional[int] = None,
                          problem_id: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate a reasoning trace for a given question.
        
        Args:
            question: The question to generate reasoning for
            max_extensions: Maximum number of reasoning extensions
            target_token_count: Target token count for reasoning
            problem_id: Optional problem ID for tracking
            **kwargs: Additional parameters for the model
            
        Returns:
            Dictionary containing the generated reasoning, answer, and metadata
        """
        # Use default values from config if not provided
        max_extensions = max_extensions or self.reasoning_config.get("max_extensions", 3)
        target_token_count = target_token_count or self.reasoning_config.get("target_token_count", 1000)
        
        # Determine if we should use think tags
        use_think_tags = self.reasoning_config.get("think_tag", False)
        
        # Prepare the system message
        system_content = self.reasoning_config.get("system_message", "You are a helpful assistant that solves problems step-by-step.")
        
        # Prepare the user message with the question
        user_content = question
        
        # Initial messages
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Start timing
        start_time = time.time()
        
        # Generate the initial response
        response = self.chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            problem_id=problem_id,
            iteration=0,
            **kwargs
        )
        
        # Extract the initial reasoning
        reasoning = response.text
        
        # Add think tags if needed
        if use_think_tags and not reasoning.startswith("<think>"):
            reasoning = f"<think>\n{reasoning}\n</think>"
        
        # Track token count (approximate)
        token_count = len(reasoning.split())
        
        # Extend reasoning if needed
        extensions = 0
        while extensions < max_extensions and token_count < target_token_count:
            # Add the assistant's response to the messages
            messages.append({"role": "assistant", "content": reasoning})
            
            # Add a continuation prompt
            continuation_prompts = self.reasoning_config.get("continuation_phrases", ["Let me continue my reasoning."])
            continuation_prompt = random.choice(continuation_prompts)
            messages.append({"role": "user", "content": continuation_prompt})
            
            # Generate the continuation
            extension_response = self.chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                problem_id=problem_id,
                iteration=extensions + 1,
                **kwargs
            )
            
            # Extract the continuation
            extension = extension_response.text
            
            # Add think tags if needed
            if use_think_tags and not extension.startswith("<think>"):
                extension = f"<think>\n{extension}\n</think>"
            
            # Append the continuation to the reasoning
            reasoning += f"\n{extension}"
            
            # Update token count
            token_count = len(reasoning.split())
            
            # Increment extensions counter
            extensions += 1
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Extract answer if possible
        answer = self.extract_answer(reasoning)
        
        # Return the result
        return {
            "reasoning": reasoning,
            "answer": answer,
            "extensions": extensions,
            "estimated_token_count": token_count,
            "generation_time": generation_time
        }
    
    def summarize_reasoning(self, reasoning_trace: str, **kwargs) -> str:
        """
        Use the model to summarize a reasoning trace.
        
        Args:
            reasoning_trace: The reasoning trace to summarize
            **kwargs: Additional model-specific parameters
            
        Returns:
            Summarized reasoning trace
        """
        # Try chat completion first if supported
        try:
            # Use chat completion for summarization (better for instruction following)
            messages = [
                {"role": "system", "content": "You are an expert at summarizing mathematical reasoning traces."},
                {"role": "user", "content": f"Summarize the following reasoning trace into a concise form that preserves all key steps and important calculations:\n\n{reasoning_trace}"}
            ]
            
            response = self.chat_completion(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", min(1000, self.max_tokens // 2)),
                temperature=kwargs.get("temperature", 0.5)  # Lower temperature for more focused summary
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"Chat completion failed for summarization, falling back to completion API: {str(e)}")
            
            # Fall back to completion API
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

    def extract_answer(self, text: str, flexible_extraction: bool = True) -> str:
        """
        Extract the answer from model output, looking for boxed notation and other patterns.
        
        Args:
            text: Text containing the answer
            flexible_extraction: If True, use more flexible patterns to extract answers
            
        Returns:
            Extracted answer or empty string if not found
        """
        if not text:
            return ""
        
        # Look for \boxed{X} pattern (LaTeX)
        pattern = r'\\boxed\{(.*?)\}'
        matches = re.findall(pattern, text)
        
        if matches:
            return matches[0].strip()
            
        # If flexible extraction is enabled, try additional patterns
        if flexible_extraction:
            # Look for "Therefore, the answer is X" pattern
            therefore_pattern = r'[Tt]herefore,?\s+the\s+answer\s+is\s*:?\s*(.*?)(?:\.|$)'
            therefore_matches = re.findall(therefore_pattern, text)
            if therefore_matches:
                return therefore_matches[0].strip()
                
            # Look for "The answer is X" pattern
            answer_pattern = r'[Tt]he\s+answer\s+is\s*:?\s*(.*?)(?:\.|$)'
            answer_matches = re.findall(answer_pattern, text)
            if answer_matches:
                return answer_matches[0].strip()
                
            # Look for "**Answer:** X" pattern (markdown)
            bold_pattern = r'\*\*Answer:\*\*\s*(.*?)(?:\.|$)'
            bold_matches = re.findall(bold_pattern, text)
            if bold_matches:
                return bold_matches[0].strip()
                
            # Look for "Answer: X" pattern
            simple_pattern = r'Answer:\s*(.*?)(?:\.|$)'
            simple_matches = re.findall(simple_pattern, text)
            if simple_matches:
                return simple_matches[0].strip()
                
            # Look for a number followed by units
            units_pattern = r'(\d+)\s*(?:miles|km|meters|m|feet|ft|pounds|lbs|kg)'
            units_matches = re.findall(units_pattern, text)
            if units_matches:
                return units_matches[0].strip()
                
            # Try to find the last equation with "=" in the text
            lines = text.split('\n')
            for line in reversed(lines):
                if '=' in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        return parts[-1].strip()
                        
            # Last resort: try to find any number in the text
            number_pattern = r'(\d+)'
            number_matches = re.findall(number_pattern, text)
            if number_matches:
                return number_matches[-1].strip()  # Take the last number
        
        return ""