#!/usr/bin/env python
"""
Debug script to identify where the 'too many values to unpack' error is occurring.
"""
import asyncio
import logging
import sys
import traceback
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the necessary modules
from src.llm.fireworks_client import FireworksModelClient
from src.reasoning.summarizer import summarize_reasoning_async
# Import TokenUsage and CostInfo from their actual location
from src.llm.fireworks_client import TokenUsage, CostInfo

async def test_generate_response_async():
    """Test the generate_response_async method of FireworksModelClient."""
    try:
        # Initialize the model client
        model = FireworksModelClient("accounts/fireworks/models/deepseek-v3-0324")
        
        # Generate a response
        prompt = "Hello, how are you?"
        response = await model.generate_response_async(
            prompt,
            max_tokens=100,
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            verbose=True
        )
        
        # Print the response type and value
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response value: {response}")
        
        # Try to unpack the response in different ways
        try:
            content, finish_reason = response
            logger.info("Successfully unpacked as (content, finish_reason)")
        except ValueError:
            logger.error("Failed to unpack as (content, finish_reason)")
        
        try:
            content, finish_reason, token_usage = response
            logger.info("Successfully unpacked as (content, finish_reason, token_usage)")
        except ValueError:
            logger.error("Failed to unpack as (content, finish_reason, token_usage)")
        
        try:
            content, finish_reason, token_usage, cost_info = response
            logger.info("Successfully unpacked as (content, finish_reason, token_usage, cost_info)")
        except ValueError:
            logger.error("Failed to unpack as (content, finish_reason, token_usage, cost_info)")
        
    except Exception as e:
        logger.error(f"Error in test_generate_response_async: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")

async def test_summarize_reasoning_async():
    """Test the summarize_reasoning_async function."""
    try:
        # Initialize the model client
        model = FireworksModelClient("accounts/fireworks/models/deepseek-v3-0324")
        
        # Generate a summary
        question = "What is 2+2?"
        reasoning = "I think 2+2 is 4."
        prompt_template = "Summarize the following reasoning: {reasoning}"
        
        summary_response = await summarize_reasoning_async(
            question,
            reasoning,
            model,
            prompt_template,
            max_tokens=100,
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            verbose=True
        )
        
        # Print the response type and value
        logger.info(f"Summary response type: {type(summary_response)}")
        logger.info(f"Summary response value: {summary_response}")
        
        # Try to unpack the summary response in different ways
        try:
            summary, summary_finish_reason = summary_response
            logger.info("Successfully unpacked as (summary, summary_finish_reason)")
        except ValueError:
            logger.error("Failed to unpack as (summary, summary_finish_reason)")
        
        try:
            summary, summary_finish_reason, token_usage = summary_response
            logger.info("Successfully unpacked as (summary, summary_finish_reason, token_usage)")
        except ValueError:
            logger.error("Failed to unpack as (summary, summary_finish_reason, token_usage)")
        
        try:
            summary, summary_finish_reason, token_usage, cost_info = summary_response
            logger.info("Successfully unpacked as (summary, summary_finish_reason, token_usage, cost_info)")
        except ValueError:
            logger.error("Failed to unpack as (summary, summary_finish_reason, token_usage, cost_info)")
        
    except Exception as e:
        logger.error(f"Error in test_summarize_reasoning_async: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")

async def main():
    """Run the tests."""
    logger.info("Testing generate_response_async...")
    await test_generate_response_async()
    
    logger.info("\nTesting summarize_reasoning_async...")
    await test_summarize_reasoning_async()

if __name__ == "__main__":
    asyncio.run(main())
