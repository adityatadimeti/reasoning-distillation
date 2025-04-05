import os
import time
import asyncio
import logging
import argparse
import numpy as np
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

# Test message for API calls - using a short prompt to minimize generation time
TEST_MESSAGE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2? Answer with just the number."}
]

# Model names for both providers
TOGETHER_MODEL = "deepseek-ai/DeepSeek-V3"
FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v3-0324"

def test_rate_limit(provider, concurrency, num_requests):
    """Test rate limits by gradually increasing concurrency."""
    logger.info(f"Testing {provider} with concurrency={concurrency}, num_requests={num_requests}")
    
    # Create client based on provider
    if provider == "together":
        client = TogetherModelClient(TOGETHER_MODEL)
    else:  # fireworks
        client = FireworksModelClient(FIREWORKS_MODEL)
    
    # Function to make a single API call
    def make_api_call(i):
        try:
            start_time = time.time()
            response = client.generate_completion(
                messages=TEST_MESSAGE,
                max_tokens=10,  # Keep this small to minimize generation time
                temperature=0.0,  # Deterministic to minimize generation time
                top_p=1.0,
                top_k=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stream=False
            )
            elapsed = time.time() - start_time
            logger.info(f"Request {i} completed in {elapsed:.2f}s")
            return True, elapsed
        except Exception as e:
            logger.error(f"Request {i} failed: {str(e)}")
            return False, 0
    
    # Execute requests in parallel
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(make_api_call, range(num_requests)))
    total_time = time.time() - start_time
    
    # Calculate statistics
    success_results = [elapsed for success, elapsed in results if success and elapsed > 0]
    success_count = len(success_results)
    
    if success_count > 0:
        avg_time = sum(success_results) / success_count
        min_time = min(success_results)
        max_time = max(success_results)
        p50 = np.percentile(success_results, 50)
        p90 = np.percentile(success_results, 90)
        p95 = np.percentile(success_results, 95)
        
        # Calculate requests per minute
        if total_time > 0:
            rpm = (success_count / total_time) * 60
        else:
            rpm = 0
    else:
        avg_time = min_time = max_time = p50 = p90 = p95 = rpm = 0
    
    # Report results
    logger.info(f"{provider.capitalize()} results:")
    logger.info(f"  Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Estimated RPM: {rpm:.1f}")
    logger.info(f"  Avg time per request: {avg_time:.2f}s")
    logger.info(f"  Min/Max time: {min_time:.2f}s / {max_time:.2f}s")
    logger.info(f"  p50/p90/p95: {p50:.2f}s / {p90:.2f}s / {p95:.2f}s")
    
    return {
        "provider": provider,
        "concurrency": concurrency,
        "success_count": success_count,
        "total_requests": num_requests,
        "success_rate": success_count/num_requests if num_requests > 0 else 0,
        "total_time": total_time,
        "rpm": rpm,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "p50": p50,
        "p90": p90,
        "p95": p95
    }

def find_max_concurrency(provider, start_concurrency=1, max_concurrency=50, target_success_rate=0.9, num_requests=10):
    """Find the maximum concurrency that maintains a target success rate."""
    logger.info(f"Finding maximum concurrency for {provider} (target success rate: {target_success_rate*100}%)")
    
    current_concurrency = start_concurrency
    max_successful_concurrency = 0
    results = []
    
    while current_concurrency <= max_concurrency:
        logger.info(f"Testing {provider} with concurrency={current_concurrency}")
        result = test_rate_limit(provider, current_concurrency, num_requests)
        results.append(result)
        
        if result["success_rate"] >= target_success_rate:
            max_successful_concurrency = current_concurrency
            # Exponential increase while successful
            current_concurrency = current_concurrency * 2
        else:
            # If we've already found a successful concurrency, do binary search
            if max_successful_concurrency > 0:
                # Binary search between last successful and current failed
                next_concurrency = (max_successful_concurrency + current_concurrency) // 2
                
                # If we're already adjacent to the max successful, we're done
                if next_concurrency == max_successful_concurrency:
                    break
                
                current_concurrency = next_concurrency
            else:
                # If we failed on the first try, try a lower concurrency
                if current_concurrency > 1:
                    current_concurrency = current_concurrency // 2
                else:
                    # Even concurrency=1 failed, we're done
                    break
    
    # Final confirmation run at the determined max concurrency
    if max_successful_concurrency > 0:
        logger.info(f"Confirming max concurrency for {provider}: {max_successful_concurrency}")
        final_result = test_rate_limit(provider, max_successful_concurrency, num_requests * 2)
        results.append(final_result)
        
        if final_result["success_rate"] >= target_success_rate:
            logger.info(f"Confirmed max concurrency for {provider}: {max_successful_concurrency}")
        else:
            logger.info(f"Final confirmation failed, adjusting max concurrency for {provider}")
            max_successful_concurrency -= 1
    
    logger.info(f"Maximum concurrency for {provider} with {target_success_rate*100}% success rate: {max_successful_concurrency}")
    return max_successful_concurrency, results

def main():
    parser = argparse.ArgumentParser(description='Compare rate limits between Together and Fireworks')
    parser.add_argument('--provider', choices=['together', 'fireworks', 'both'], default='both',
                        help='Which provider to test')
    parser.add_argument('--concurrency', type=int, default=0, 
                        help='Fixed concurrency to test (0 to auto-detect max concurrency)')
    parser.add_argument('--num-requests', type=int, default=20, 
                        help='Number of requests per test')
    parser.add_argument('--target-success-rate', type=float, default=0.9,
                        help='Target success rate for auto-detection (0.0-1.0)')
    args = parser.parse_args()
    
    providers = []
    if args.provider == 'both':
        providers = ['together', 'fireworks']
    else:
        providers = [args.provider]
    
    results = {}
    
    for provider in providers:
        if args.concurrency > 0:
            # Fixed concurrency test
            result = test_rate_limit(provider, args.concurrency, args.num_requests)
            results[provider] = [result]
        else:
            # Auto-detect maximum concurrency
            max_concurrency, provider_results = find_max_concurrency(
                provider, 
                start_concurrency=1,
                max_concurrency=50,
                target_success_rate=args.target_success_rate,
                num_requests=args.num_requests
            )
            results[provider] = provider_results
    
    # Final comparison
    if len(providers) > 1:
        logger.info("\n=== FINAL COMPARISON ===")
        for provider in providers:
            # Find the result with the highest concurrency that met the target success rate
            successful_results = [r for r in results[provider] if r["success_rate"] >= args.target_success_rate]
            if successful_results:
                best_result = max(successful_results, key=lambda r: r["concurrency"])
                logger.info(f"{provider.capitalize()}:")
                logger.info(f"  Max concurrency: {best_result['concurrency']}")
                logger.info(f"  Success rate: {best_result['success_rate']*100:.1f}%")
                logger.info(f"  Estimated RPM: {best_result['rpm']:.1f}")
            else:
                logger.info(f"{provider.capitalize()}: No successful tests")

if __name__ == "__main__":
    main()
