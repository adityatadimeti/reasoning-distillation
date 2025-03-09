import os
import threading
import webbrowser
import logging
import json
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
        self.latest_status = None  # Store the latest experiment status
        self.setup_routes()
        self.setup_socketio()
        
    def setup_routes(self):
        """Set up Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
            
        @self.app.route('/view_results/<path:result_path>')
        def view_results(result_path):
            """Load saved experiment results"""
            logger.info(f"View results route accessed with path: {result_path}")
            try:
                # Check if path is already absolute (could happen with redirect from index)
                if os.path.isabs(result_path):
                    full_path = result_path
                else:
                    # If it's relative, make it absolute
                    full_path = os.path.join(os.getcwd(), result_path)
                
                logger.info(f"Resolved path: {full_path}")
                
                # Check if it's a directory or direct path to results.json
                if os.path.isdir(full_path):
                    results_file = os.path.join(full_path, "results.json")
                else:
                    results_file = full_path
                
                if not os.path.exists(results_file):
                    logger.error(f"Results file not found: {results_file}")
                    return render_template('dashboard.html', error=f"Results file not found: {results_file}")
                
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                
                # Log loaded results
                logger.info(f"Loaded static results with {len(results_data.get('results', []))} problems")
                
                # Don't convert to JSON string here - let Flask/Jinja handle it with the tojson filter
                logger.info(f"Passing static_results to template")
                logger.info(f"static_mode flag set to: True")
                
                # Create a debug version of the template context
                template_context = {
                    'static_results': results_data,  # Pass the Python object, not a JSON string
                    'static_mode': True
                }
                logger.info(f"Template context keys: {list(template_context.keys())}")
                
                # Pass the results data to the template with explicit static mode flag
                rendered = render_template('dashboard.html', **template_context)
                logger.info(f"Template rendered, size: {len(rendered)} bytes")
                return rendered
            except Exception as e:
                return render_template('dashboard.html', error=f"Error loading results: {str(e)}")
    
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
            
        @self.socketio.on('request_current_status')
        def handle_request_status():
            logger.info("Client requested current experiment status")
            if self.latest_status:
                logger.info(f"Resending latest status: {self.latest_status.get('status', 'unknown')}")
                emit('experiment_status', self.latest_status)
    
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
    
    @staticmethod
    def static_start(results_path: str, port: int = 8080, open_browser: bool = True):
        """
        Start the dashboard server in static mode to view previous experiment results.
        
        Args:
            results_path: Path to results.json file or directory containing results.json
            port: Port to run the server on
            open_browser: Whether to automatically open the dashboard in a browser
        """
        logger.info(f"Static start method called with path: {results_path}")
        
        app = Flask(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"),
        )
        
        @app.route('/')
        def index():
            # Get the absolute path but use only the final part after the current working directory
            cwd = os.getcwd()
            abs_path = os.path.abspath(results_path)
            
            # If the path is within the current directory, make it relative
            if abs_path.startswith(cwd):
                rel_path = os.path.relpath(abs_path, cwd)
            else:
                rel_path = abs_path
                
            logger.info(f"Creating redirect with relative path: {rel_path}")
            return f'<script>window.location.href = "/view_results/{rel_path}";</script>'
        
        @app.route('/view_results/<path:result_path>')
        def view_results(result_path):
            """Load saved experiment results"""
            logger.info(f"View results route accessed with path: {result_path}")
            try:
                # Check if path is already absolute (could happen with redirect from index)
                if os.path.isabs(result_path):
                    full_path = result_path
                else:
                    # If it's relative, make it absolute
                    full_path = os.path.join(os.getcwd(), result_path)
                
                logger.info(f"Resolved path: {full_path}")
                
                # Check if it's a directory or direct path to results.json
                if os.path.isdir(full_path):
                    results_file = os.path.join(full_path, "results.json")
                else:
                    results_file = full_path
                
                if not os.path.exists(results_file):
                    logger.error(f"Results file not found: {results_file}")
                    return render_template('dashboard.html', error=f"Results file not found: {results_file}")
                
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                
                # Log loaded results
                logger.info(f"Loaded static results with {len(results_data.get('results', []))} problems")
                
                # Don't convert to JSON string here - let Flask/Jinja handle it with the tojson filter
                logger.info(f"Passing static_results to template")
                logger.info(f"static_mode flag set to: True")
                
                # Create a debug version of the template context
                template_context = {
                    'static_results': results_data,  # Pass the Python object, not a JSON string
                    'static_mode': True
                }
                logger.info(f"Template context keys: {list(template_context.keys())}")
                
                # Pass the results data to the template with explicit static mode flag
                rendered = render_template('dashboard.html', **template_context)
                logger.info(f"Template rendered, size: {len(rendered)} bytes")
                return rendered
            except Exception as e:
                return render_template('dashboard.html', error=f"Error loading results: {str(e)}")
                
        if open_browser:
            webbrowser.open(f'http://localhost:{port}')
            
        logger.info(f"Starting static dashboard on http://localhost:{port}")
        logger.info(f"Viewing results from: {results_path}")
        
        app.run(host='0.0.0.0', port=port, debug=False)
    
    def update_experiment_status(self, data: Dict[str, Any]):
        """
        Update experiment status on the dashboard.
        
        Args:
            data: Status data to send to the dashboard
        """
        # Store the latest status
        self.latest_status = data
        
        if self.thread and self.thread.is_alive():
            has_config = 'config' in data
            logger.info(f"Emitting experiment_status event. Status: {data.get('status')}, Has config: {has_config}")
            self.socketio.emit('experiment_status', data)
    
    def update_problem_status(self, problem_id: str, status: str, question: str = None):
        """
        Update status for a specific problem.
        
        Args:
            problem_id: ID of the problem
            status: Status message
            question: Optional question text
        """
        if self.thread and self.thread.is_alive():
            data = {
                'problem_id': problem_id,
                'status': status
            }
            
            # Include question text if provided
            if question is not None:
                data['question'] = question
                
            self.socketio.emit('problem_status', data)
    
    def stream_model_output(self, problem_id: str, chunk: str, iteration: int = 0):
        """
        Stream a chunk of model output to the dashboard.
        
        Args:
            problem_id: ID of the problem
            chunk: Text chunk from the model
            iteration: Iteration number (0 = initial, 1 = first improvement, etc.)
        """
        if self.thread and self.thread.is_alive():
            self.socketio.emit('model_output', {
                'problem_id': problem_id,
                'chunk': chunk,
                'iteration': iteration
            })
    
    def update_answer_info(self, problem_id: str, answer: str, correct_answer: str, is_correct: bool, iteration: int = 0):
        """
        Update answer information for a problem.
        
        Args:
            problem_id: ID of the problem
            answer: The extracted answer
            correct_answer: The correct answer
            is_correct: Whether the answer is correct
            iteration: Iteration number
        """
        if self.thread and self.thread.is_alive():
            self.socketio.emit('answer_info', {
                'problem_id': problem_id,
                'answer': answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'iteration': iteration
            })
    
    def update_summary(self, problem_id: str, summary: str, iteration: int = 0):
        """
        Update summary for a problem.
        
        Args:
            problem_id: ID of the problem
            summary: The summary text
            iteration: The iteration this summary belongs to
        """
        if self.thread and self.thread.is_alive():
            self.socketio.emit('summary', {
                'problem_id': problem_id,
                'summary': summary,
                'iteration': iteration
            })
    
    def stream_summary_chunk(self, problem_id: str, chunk: str, iteration: int = 0):
        """
        Stream a chunk of summary text to the dashboard.
        
        Args:
            problem_id: ID of the problem
            chunk: Text chunk of the summary
            iteration: The iteration this summary belongs to
        """
        if self.thread and self.thread.is_alive():
            self.socketio.emit('summary_chunk', {
                'problem_id': problem_id,
                'chunk': chunk,
                'iteration': iteration
            })
    
    def stop(self):
        """Stop the dashboard server."""
        if self.thread and self.thread.is_alive():
            logger.info("Shutting down dashboard server")
            # Stopping Flask in a thread is tricky, but for our purposes
            # we can rely on the daemon thread being killed when the main process exits 