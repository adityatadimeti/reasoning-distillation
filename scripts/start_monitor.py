#!/usr/bin/env python
"""
Script to start the experiment monitor server separately.
"""
import os
import sys
import logging
import time
import threading

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.experiment_monitor import start_monitor_server, load_experiment_state

def periodic_state_reload():
    """Periodically reload the experiment state from file."""
    while True:
        load_experiment_state()
        time.sleep(2)  # Reload every 2 seconds

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the monitor server
    port = 5001  # Use a different port than the default 5000
    print(f"Starting experiment monitor server on http://localhost:{port}")
    if not start_monitor_server(port=port):
        print("Failed to start monitor server")
        sys.exit(1)
    
    # Start a thread to periodically reload the experiment state
    reload_thread = threading.Thread(target=periodic_state_reload)
    reload_thread.daemon = True
    reload_thread.start()
    
    # Keep the script running
    print("Monitor server started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Monitor server stopped.") 