import os
import threading
import webbrowser
import logging
from typing import Dict, Any
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

class DashboardServer:
    """Simple dashboard server for real-time experiment progress tracking."""
    
    def __init__(self, port: int = 8080):
        """
        Initialize the dashboard server.
        
        Args:
            port: Port to run the server on
        """
        self.port = port
        self.app = Flask(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"),
        )
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.thread = None
        self.client_ready = False
        self.setup_routes()
        self.setup_socketio()
        
    def setup_routes(self):
        """Set up Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
    
    def setup_socketio(self):
        """Set up SocketIO event handlers."""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected to dashboard")
            emit('status', {'message': 'Connected to experiment server'})
            
        @self.socketio.on('client_ready')
        def handle_client_ready():
            logger.info("Client ready to receive data")
            self.client_ready = True
    
    def start(self, open_browser: bool = True):
        """
        Start the dashboard server in a separate thread.
        
        Args:
            open_browser: Whether to automatically open the dashboard in a browser
        """
        def run_server():
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)
        
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=run_server)
            self.thread.daemon = True
            self.thread.start()
            
            if open_browser:
                webbrowser.open(f'http://localhost:{self.port}')
            
            logger.info(f"Dashboard server started on http://localhost:{self.port}")
    
    def update_experiment_status(self, data: Dict[str, Any]):
        """
        Update experiment status on the dashboard.
        
        Args:
            data: Status data to send to the dashboard
        """
        if self.thread and self.thread.is_alive():
            self.socketio.emit('experiment_status', data)
    
    def update_problem_status(self, problem_id: str, status: str):
        """
        Update status for a specific problem.
        
        Args:
            problem_id: ID of the problem
            status: Status message
        """
        if self.thread and self.thread.is_alive():
            self.socketio.emit('problem_status', {
                'problem_id': problem_id,
                'status': status
            })
    
    def stream_model_output(self, problem_id: str, chunk: str):
        """
        Stream a chunk of model output to the dashboard.
        
        Args:
            problem_id: ID of the problem
            chunk: Text chunk from the model
        """
        logger.debug(f"Streaming chunk to dashboard for problem ID: {problem_id}")
        if self.thread and self.thread.is_alive() and self.client_ready:
            self.socketio.emit('model_output', {
                'problem_id': problem_id,
                'chunk': chunk
            })
    
    def update_answer_info(self, problem_id: str, extracted_answer: str, correct_answer: str, is_correct: bool):
        """
        Send answer information to the dashboard.
        
        Args:
            problem_id: ID of the problem
            extracted_answer: The answer extracted from the model's reasoning
            correct_answer: The correct answer to the problem
            is_correct: Whether the extracted answer matches the correct answer
        """
        logger.debug(f"Sending answer info to dashboard for problem ID: {problem_id}")
        if self.thread and self.thread.is_alive() and self.client_ready:
            self.socketio.emit('answer_info', {
                'problem_id': problem_id,
                'extracted_answer': extracted_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct
            })
    
    def stop(self):
        """Stop the dashboard server."""
        if self.thread and self.thread.is_alive():
            logger.info("Shutting down dashboard server")
            # Stopping Flask in a thread is tricky, but for our purposes
            # we can rely on the daemon thread being killed when the main process exits 