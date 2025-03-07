"""
Experiment monitoring utilities for tracking and displaying experiment progress.
"""
import os
import json
import time
import threading
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from queue import Queue, Empty
from flask import Flask, render_template, jsonify, request

logger = logging.getLogger(__name__)

# Global variables to store experiment state
current_experiment = {
    "status": "idle",
    "pipeline": None,
    "problems_total": 0,
    "problems_processed": 0,
    "current_problem": None,
    "current_iteration": 0,
    "iterations_total": 0,
    "reasoning_traces": [],
    "api_messages": [],
    "start_time": None,
    "elapsed_time": 0
}

# Queue for passing messages between experiment and monitor
message_queue = Queue()

# Path for storing experiment state
EXPERIMENT_STATE_FILE = os.path.join(os.path.dirname(__file__), "experiment_state.json")

def save_experiment_state():
    """Save the current experiment state to a file."""
    try:
        with open(EXPERIMENT_STATE_FILE, 'w') as f:
            # Create a copy of the state that's JSON serializable
            state_copy = current_experiment.copy()
            # Convert timestamps to strings if needed
            if state_copy["start_time"] is not None:
                state_copy["start_time"] = str(state_copy["start_time"])
            json.dump(state_copy, f)
        logger.debug(f"Saved experiment state to {EXPERIMENT_STATE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save experiment state: {str(e)}")

def load_experiment_state():
    """Load the experiment state from a file."""
    global current_experiment
    try:
        if os.path.exists(EXPERIMENT_STATE_FILE):
            with open(EXPERIMENT_STATE_FILE, 'r') as f:
                state = json.load(f)
                # Convert string timestamps back to float if needed
                if state.get("start_time") and isinstance(state["start_time"], str):
                    try:
                        state["start_time"] = float(state["start_time"])
                    except ValueError:
                        state["start_time"] = None
                current_experiment.update(state)
            logger.debug(f"Loaded experiment state from {EXPERIMENT_STATE_FILE}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to load experiment state: {str(e)}")
        return False

# Flask app for the web interface
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

def initialize_monitor(pipeline_name: str, total_problems: int, max_iterations: int):
    """
    Initialize the experiment monitor with basic information.
    
    Args:
        pipeline_name: Name of the pipeline being run
        total_problems: Total number of problems to process
        max_iterations: Maximum number of iterations per problem
    """
    global current_experiment
    current_experiment["status"] = "running"
    current_experiment["pipeline"] = pipeline_name
    current_experiment["problems_total"] = total_problems
    current_experiment["iterations_total"] = max_iterations
    current_experiment["problems_processed"] = 0
    current_experiment["current_problem"] = None
    current_experiment["current_iteration"] = 0
    current_experiment["reasoning_traces"] = []
    current_experiment["api_messages"] = []
    current_experiment["start_time"] = time.time()
    current_experiment["elapsed_time"] = 0
    
    # Save the state to file
    save_experiment_state()
    
    # Send initialization message to queue
    message_queue.put({
        "type": "init",
        "data": {
            "pipeline": pipeline_name,
            "total_problems": total_problems,
            "max_iterations": max_iterations
        }
    })

def update_problem_status(problem_id: str, question: str, iteration: int):
    """
    Update the status for the current problem being processed.
    
    Args:
        problem_id: ID of the current problem
        question: Question text
        iteration: Current iteration number
    """
    global current_experiment
    current_experiment["current_problem"] = {
        "id": problem_id,
        "question": question
    }
    current_experiment["current_iteration"] = iteration
    
    # Handle case where start_time might be None
    if current_experiment["start_time"] is None:
        current_experiment["start_time"] = time.time()
        current_experiment["elapsed_time"] = 0
    else:
        current_experiment["elapsed_time"] = time.time() - current_experiment["start_time"]
    
    # Save the state to file
    save_experiment_state()
    
    # Send problem update message to queue
    message_queue.put({
        "type": "problem_update",
        "data": {
            "problem_id": problem_id,
            "question": question,
            "iteration": iteration
        }
    })

def add_reasoning_trace(problem_id: str, iteration: int, reasoning: str, answer: str = None,
                   confidence: float = None, errors_detected: bool = None, should_continue: bool = None):
    """
    Add a reasoning trace for the current problem and iteration.
    
    Args:
        problem_id: ID of the current problem
        iteration: Iteration number
        reasoning: Reasoning trace text
        answer: Extracted answer (if available)
        confidence: Confidence score for the answer
        errors_detected: Whether errors were detected in the reasoning
        should_continue: Whether the reasoning should continue
    """
    global current_experiment
    trace = {
        "problem_id": problem_id,
        "iteration": iteration,
        "reasoning": reasoning,
        "answer": answer,
        "confidence": confidence,
        "errors_detected": errors_detected,
        "should_continue": should_continue,
        "timestamp": time.time()
    }
    current_experiment["reasoning_traces"].append(trace)
    
    # Save the state to file
    save_experiment_state()
    
    # Send reasoning trace message to queue
    message_queue.put({
        "type": "reasoning_trace",
        "data": trace
    })

def add_api_message(problem_id: str, iteration: int, direction: str, messages: List[Dict[str, Any]]):
    """
    Add an API message exchange for tracking.
    
    Args:
        problem_id: ID of the current problem
        iteration: Iteration number
        direction: "request" or "response"
        messages: List of message dictionaries
    """
    global current_experiment
    message_data = {
        "problem_id": problem_id,
        "iteration": iteration,
        "direction": direction,
        "messages": messages,
        "timestamp": time.time()
    }
    current_experiment["api_messages"].append(message_data)
    
    # Send API message to queue
    message_queue.put({
        "type": "api_message",
        "data": message_data
    })

def complete_problem(problem_id: str, final_answer: str, processing_time: float = None, iteration_count: int = None):
    """
    Mark a problem as completed.
    
    Args:
        problem_id: ID of the completed problem
        final_answer: Final answer for the problem
        processing_time: Time taken to process the problem (in seconds)
        iteration_count: Number of iterations used for the problem
    """
    global current_experiment
    current_experiment["problems_processed"] += 1
    
    # Save the state to file
    save_experiment_state()
    
    # Send problem completion message to queue
    message_queue.put({
        "type": "problem_complete",
        "data": {
            "problem_id": problem_id,
            "final_answer": final_answer,
            "problems_processed": current_experiment["problems_processed"],
            "processing_time": processing_time,
            "iteration_count": iteration_count
        }
    })

def complete_experiment(metrics: Dict[str, Any] = None):
    """
    Mark the experiment as completed.
    
    Args:
        metrics: Dictionary of experiment metrics
    """
    global current_experiment
    current_experiment["status"] = "completed"
    
    # Handle case where start_time might be None
    if current_experiment["start_time"] is None:
        current_experiment["start_time"] = time.time()
        current_experiment["elapsed_time"] = 0
    else:
        current_experiment["elapsed_time"] = time.time() - current_experiment["start_time"]
        
    current_experiment["metrics"] = metrics
    
    # Save the state to file
    save_experiment_state()
    
    # Send experiment completion message to queue
    message_queue.put({
        "type": "experiment_complete",
        "data": {
            "elapsed_time": current_experiment["elapsed_time"],
            "metrics": metrics
        }
    })

def start_monitor_server(host: str = "localhost", port: int = 5000):
    """
    Start the Flask server for monitoring the experiment.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """
    try:
        # Check if the templates directory exists
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        logger.info(f"Looking for templates directory at: {template_dir}")
        if not os.path.exists(template_dir):
            logger.error(f"Templates directory not found at {template_dir}")
            logger.error("Monitor server cannot start without templates")
            return False
        else:
            logger.info(f"Templates directory found at {template_dir}")
            
        # Check if the monitor.html template exists
        monitor_template = os.path.join(template_dir, "monitor.html")
        logger.info(f"Looking for monitor.html template at: {monitor_template}")
        if not os.path.exists(monitor_template):
            logger.error(f"monitor.html template not found at {monitor_template}")
            logger.error("Monitor server cannot start without monitor.html template")
            return False
        else:
            logger.info(f"monitor.html template found at {monitor_template}")
        
        # Try to load existing experiment state
        if load_experiment_state():
            logger.info("Loaded existing experiment state")
        else:
            logger.info("No existing experiment state found, starting with default state")
        
        # Try to start the server, trying different ports if necessary
        max_port_attempts = 10
        current_port = port
        
        logger.info(f"Attempting to start monitor server on {host}:{current_port}")
        for attempt in range(max_port_attempts):
            try:
                # Start the server in a separate thread
                server_thread = threading.Thread(
                    target=lambda: app.run(host=host, port=current_port, debug=False, use_reloader=False)
                )
                server_thread.daemon = True  # Make thread a daemon so it exits when main thread exits
                server_thread.start()
                
                # Wait a moment to see if the server starts successfully
                time.sleep(1.0)
                
                # Log success
                logger.info(f"Experiment monitor started at http://{host}:{current_port}")
                return True
            except Exception as e:
                if "Address already in use" in str(e):
                    logger.warning(f"Port {current_port} is already in use, trying port {current_port + 1}")
                    current_port += 1
                else:
                    logger.error(f"Failed to start monitor server: {str(e)}")
                    return False
        
        # If we've tried all ports and failed
        logger.error(f"Failed to find an available port after {max_port_attempts} attempts")
        return False
            
    except Exception as e:
        logger.error(f"Failed to start monitor server: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('monitor.html')

@app.route('/api/status')
def get_status():
    """API endpoint to get the current experiment status."""
    global current_experiment
    
    # Load the latest state from file
    load_experiment_state()
    
    # Update elapsed time if experiment is running
    if current_experiment["status"] == "running" and current_experiment["start_time"] is not None:
        current_experiment["elapsed_time"] = time.time() - current_experiment["start_time"]
    
    # Add information about available iterations
    if current_experiment["current_problem"] and current_experiment["reasoning_traces"]:
        problem_id = current_experiment["current_problem"]["id"]
        available_iterations = [
            trace["iteration"] for trace in current_experiment["reasoning_traces"]
            if trace["problem_id"] == problem_id
        ]
        current_experiment["available_iterations"] = sorted(list(set(available_iterations)))
    else:
        current_experiment["available_iterations"] = []
    
    return jsonify(current_experiment)

@app.route('/api/reasoning/<problem_id>/<int:iteration>')
def get_reasoning(problem_id, iteration):
    """API endpoint to get a specific reasoning trace."""
    global current_experiment
    
    # Load the latest state from file
    load_experiment_state()
    
    for trace in current_experiment["reasoning_traces"]:
        if trace["problem_id"] == problem_id and trace["iteration"] == iteration:
            return jsonify(trace)
    return jsonify({"error": "Reasoning trace not found"}), 404

@app.route('/api/messages/<problem_id>/<int:iteration>')
def get_messages(problem_id, iteration):
    """API endpoint to get API messages for a specific problem and iteration."""
    global current_experiment
    
    # Load the latest state from file
    load_experiment_state()
    
    messages = [msg for msg in current_experiment["api_messages"] 
                if msg["problem_id"] == problem_id and msg["iteration"] == iteration]
    return jsonify(messages)

@app.route('/api/updates', methods=['GET'])
def get_updates():
    """API endpoint for long-polling updates."""
    try:
        # Wait for a message with a timeout
        message = message_queue.get(timeout=30)
        return jsonify(message)
    except Empty:
        # Return an empty update if no message is available
        return jsonify({"type": "heartbeat"}) 