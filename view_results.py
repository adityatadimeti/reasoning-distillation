#!/usr/bin/env python
"""
Script to view experiment results in a static dashboard.
"""

import argparse
import logging
import os
import json
from src.dashboard.server import DashboardServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="View experiment results in a static dashboard")
    parser.add_argument(
        "results_path", 
        help="Path to results.json file or directory containing results.json"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=9080, 
        help="Port to run the dashboard on (default: 9080)"
    )
    parser.add_argument(
        "--no-browser", 
        action="store_true", 
        help="Don't open the dashboard in a browser"
    )
    
    args = parser.parse_args()
    
    # Check if the path exists
    if not os.path.exists(args.results_path):
        logger.error(f"Results path not found: {args.results_path}")
        return 1
    
    # Start the dashboard in static mode
    try:
        logger.info(f"Starting static dashboard with results path: {args.results_path}")
        logger.info(f"Absolute path: {os.path.abspath(args.results_path)}")
        if os.path.isdir(args.results_path):
            results_file = os.path.join(args.results_path, "results.json")
        else:
            results_file = args.results_path
        logger.info(f"Results file: {results_file}")
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                try:
                    data = json.load(f)
                    logger.info(f"Successfully loaded results file with {len(data.get('results', []))} problems")
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {str(e)}")
        else:
            logger.error(f"Results file not found: {results_file}")
        
        DashboardServer.static_start(
            results_path=args.results_path,
            port=args.port,
            open_browser=not args.no_browser
        )
    except KeyboardInterrupt:
        logger.info("Dashboard server stopped by user")
    except Exception as e:
        logger.error(f"Error starting dashboard: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 