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
    current_experiment["elapsed_time"] = time.time() - current_experiment["start_time"]
    
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
    current_experiment["elapsed_time"] = time.time() - current_experiment["start_time"]
    current_experiment["metrics"] = metrics
    
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
    threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False)).start()
    logger.info(f"Experiment monitor started at http://{host}:{port}")

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('monitor.html')

@app.route('/api/status')
def get_status():
    """API endpoint to get the current experiment status."""
    global current_experiment
    return jsonify(current_experiment)

@app.route('/api/reasoning/<problem_id>/<int:iteration>')
def get_reasoning(problem_id, iteration):
    """API endpoint to get a specific reasoning trace."""
    global current_experiment
    for trace in current_experiment["reasoning_traces"]:
        if trace["problem_id"] == problem_id and trace["iteration"] == iteration:
            return jsonify(trace)
    return jsonify({"error": "Reasoning trace not found"}), 404

@app.route('/api/messages/<problem_id>/<int:iteration>')
def get_messages(problem_id, iteration):
    """API endpoint to get API messages for a specific problem and iteration."""
    global current_experiment
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