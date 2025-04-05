import os
import time
import asyncio
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from src.llm.together_client import TogetherModelClient
from src.llm.fireworks_client import FireworksModelClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test message for API calls
TEST_MESSAGE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

def test_together_client(concurrency=5, num_requests=10):
    """Test Together client with high concurrency to verify rate limit handling."""
    logger.info(f"Testing Together client with concurrency={concurrency}, num_requests={num_requests}")
    
    # Create a Together client with a common model
    client = TogetherModelClient("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    
    # Function to make a single API call
    def make_api_call(i):
        try:
            start_time = time.time()
            response = client.generate_completion(
                messages=TEST_MESSAGE,
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stream=False
            )
            elapsed = time.time() - start_time
            logger.info(f"Request {i} completed in {elapsed:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Request {i} failed: {str(e)}")
            return False
    
    # Execute requests in parallel
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(make_api_call, range(num_requests)))
    
    # Report results
    success_count = sum(results)
    logger.info(f"Together client: {success_count}/{num_requests} requests succeeded")
    return success_count == num_requests

def test_fireworks_client(concurrency=5, num_requests=10):
    """Test Fireworks client with high concurrency to verify rate limit handling."""
    logger.info(f"Testing Fireworks client with concurrency={concurrency}, num_requests={num_requests}")
    
    # Create a Fireworks client with a common model
    client = FireworksModelClient("accounts/fireworks/models/llama4-maverick-instruct-basic")
    
    # Function to make a single API call
    def make_api_call(i):
        try:
            start_time = time.time()
            response = client.generate_completion(
                messages=TEST_MESSAGE,
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stream=False
            )
            elapsed = time.time() - start_time
            logger.info(f"Request {i} completed in {elapsed:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Request {i} failed: {str(e)}")
            return False
    
    # Execute requests in parallel
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(make_api_call, range(num_requests)))
    
    # Report results
    success_count = sum(results)
    logger.info(f"Fireworks client: {success_count}/{num_requests} requests succeeded")
    return success_count == num_requests

async def test_together_client_async(concurrency=5, num_requests=10):
    """Test Together client with async requests to verify rate limit handling."""
    logger.info(f"Testing Together client async with concurrency={concurrency}, num_requests={num_requests}")
    
    # Create a Together client with a DeepSeek model
    client = TogetherModelClient("deepseek-ai/deepseek-llama-70b-chat")
    
    # Function to make a single API call
    async def make_api_call(i):
        try:
            start_time = time.time()
            response = await client.generate_completion_async(
                messages=TEST_MESSAGE,
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stream=False
            )
            elapsed = time.time() - start_time
            logger.info(f"Async request {i} completed in {elapsed:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Async request {i} failed: {str(e)}")
            return False
    
    # Create and gather tasks
    tasks = [make_api_call(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    # Report results
    success_count = sum(results)
    logger.info(f"Together client async: {success_count}/{num_requests} requests succeeded")
    return success_count == num_requests

async def test_fireworks_client_async(concurrency=5, num_requests=10):
    """Test Fireworks client with async requests to verify rate limit handling."""
    logger.info(f"Testing Fireworks client async with concurrency={concurrency}, num_requests={num_requests}")
    
    # Create a Fireworks client with a common model
    client = FireworksModelClient("accounts/fireworks/models/mixtral-8x7b-instruct")
    
    # Function to make a single API call
    async def make_api_call(i):
        try:
            start_time = time.time()
            response = await client.generate_completion_async(
                messages=TEST_MESSAGE,
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stream=False
            )
            elapsed = time.time() - start_time
            logger.info(f"Async request {i} completed in {elapsed:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Async request {i} failed: {str(e)}")
            return False
    
    # Create and gather tasks
    tasks = [make_api_call(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    # Report results
    success_count = sum(results)
    logger.info(f"Fireworks client async: {success_count}/{num_requests} requests succeeded")
    return success_count == num_requests

async def run_async_tests(concurrency=5, num_requests=10):
    """Run all async tests."""
    together_success = await test_together_client_async(concurrency, num_requests)
    fireworks_success = await test_fireworks_client_async(concurrency, num_requests)
    return together_success and fireworks_success

def main():
    parser = argparse.ArgumentParser(description='Test rate limit handling in API clients')
    parser.add_argument('--concurrency', type=int, default=5, 
                        help='Number of concurrent requests')
    parser.add_argument('--num-requests', type=int, default=10, 
                        help='Total number of requests to make')
    parser.add_argument('--use-async', action='store_true', 
                        help='Use async API calls instead of threaded calls')
    parser.add_argument('--provider', choices=['together', 'fireworks', 'both'], default='both',
                        help='Which provider to test')
    args = parser.parse_args()
    
    if args.provider in ['together', 'both'] and not args.use_async:
        test_together_client(args.concurrency, args.num_requests)
    
    if args.provider in ['fireworks', 'both'] and not args.use_async:
        test_fireworks_client(args.concurrency, args.num_requests)
    
    if args.use_async:
        asyncio.run(run_async_tests(args.concurrency, args.num_requests))

if __name__ == "__main__":
    main()
