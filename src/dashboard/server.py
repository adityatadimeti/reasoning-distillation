import os
import threading
import webbrowser
import logging
import json
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import sys

# Add the project root to the path so we can import run_experiment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

class DashboardServer:
    """Simple dashboard server for real-time experiment progress tracking."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, port=5000, config_path=None):
        """Get or create a singleton instance of the dashboard server."""
        if cls._instance is None:
            cls._instance = cls(port=port, config_path=config_path)
        return cls._instance
    
    def __init__(self, port=5000, config_path=None):
        """
        Initialize the dashboard server.
        
        Args:
            port: Port to run the server on
            config_path: Path to the configuration file
        """
        self.port = port
        self.config_path = config_path
        self.app = Flask(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"),
        )
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.thread = None
        self.running = False
        self.experiment_running = False
        self.latest_status = None  # Store the latest experiment status
        self.setup_routes()
        self.setup_socketio()
        
    def setup_routes(self):
        """Set up Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('dashboard.html', static_mode=False)
        
        @self.app.route('/static/<path:path>')
        def serve_static(path):
            return send_from_directory('static', path)
        
        @self.app.route('/api/start_experiment', methods=['POST'])
        def start_experiment():
            print("DEBUG: /api/start_experiment endpoint called")
            if self.experiment_running:
                print("DEBUG: Experiment already running, returning error")
                return jsonify({'status': 'error', 'message': 'Experiment already running'})
            
            # Get model parameters from request
            data = request.json or {}
            print(f"DEBUG: Received request data: {data}")
            
            # Create a model_params dictionary with all the necessary keys
            model_params = {
                'reasoning_model': data.get('reasoning_model'),
                'summarizer_model': data.get('summarizer_model'),
                'summarizer_type': data.get('summarizer_type', 'external')  # Default to external
            }
            
            # Log what we're passing to the experiment
            print(f"DEBUG: Model params to be used: {model_params}")
            
            # Start experiment in a separate thread
            print("DEBUG: Starting experiment thread")
            thread = threading.Thread(
                target=self._run_experiment,
                args=(self.config_path, model_params),
                daemon=True
            )
            thread.start()
            print("DEBUG: Experiment thread started")
            
            return jsonify({'status': 'started'})
        
        @self.app.route('/api/config', methods=['GET'])
        def get_config():
            """Get the current configuration."""
            from src.utils.config import load_config
            if not self.config_path:
                return jsonify({'status': 'error', 'message': 'No configuration file specified'})
            
            try:
                config = load_config(self.config_path)
                return jsonify({'status': 'success', 'config': config})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/reset_experiment_state', methods=['POST'])
        def reset_experiment_state():
            """Reset the experiment running state."""
            print("DEBUG: Resetting experiment running state")
            self.experiment_running = False
            return jsonify({'status': 'success', 'message': 'Experiment state reset'})
    
    def setup_socketio(self):
        """Set up SocketIO event handlers."""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected to dashboard")
            emit('status', {'message': 'Connected to experiment server'})
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Client disconnected')
            
        @self.socketio.on('client_ready')
        def handle_client_ready():
            logger.info("Client ready to receive data")
            # If we have a config, send it to the client
            if self.config_path:
                from src.utils.config import load_config
                try:
                    config = load_config(self.config_path)
                    self.update_experiment_status({
                        'experiment_name': config.get('experiment_name', 'Experiment'),
                        'status': 'Ready',
                        'config': config
                    })
                except Exception as e:
                    logger.error(f"Error loading config: {e}")
            
        @self.socketio.on('request_current_status')
        def handle_request_status():
            logger.info("Client requested current experiment status")
            if self.latest_status:
                logger.info(f"Resending latest status: {self.latest_status.get('status', 'unknown')}")
                emit('experiment_status', self.latest_status)
    
    def _run_experiment(self, config_path, model_params):
        """Run an experiment with the given configuration."""
        from run_experiment import run_experiment
        import traceback
        
        print(f"DEBUG: _run_experiment called with config_path={config_path}")
        print(f"DEBUG: Model parameters received: {model_params}")
        
        if not config_path:
            print("DEBUG: No configuration file specified")
            logger.error("No configuration file specified")
            self.socketio.emit('status', {'message': 'Error: No configuration file specified'})
            self.experiment_running = False  # Reset state
            return
        
        try:
            # Load config to see original values
            from src.utils.config import load_config
            original_config = load_config(config_path)
            print(f"DEBUG: Original config from file:")
            print(f"  - reasoning_model: {original_config.get('reasoning_model')}")
            print(f"  - summarizer_type: {original_config.get('summarizer_type')}")
            print(f"  - summarizer_model: {original_config.get('summarizer_model')}")
            
            print("DEBUG: Setting experiment_running = True")
            self.experiment_running = True
            self.socketio.emit('status', {'message': 'Experiment starting'})
            
            # Run the experiment with model overrides
            print(f"DEBUG: Calling run_experiment with model_params={model_params}")
            result = run_experiment(
                config_path=config_path,
                use_dashboard=True,
                model_params=model_params,
                verbose=True
            )
            
            # Check what model values were actually used
            if result and 'config' in result:
                used_config = result['config']
                print(f"DEBUG: Config values used for experiment:")
                print(f"  - reasoning_model: {used_config.get('reasoning_model')}")
                print(f"  - summarizer_type: {used_config.get('summarizer_type')}")  
                print(f"  - summarizer_model: {used_config.get('summarizer_model')}")
            
            print(f"DEBUG: run_experiment completed")
            
            self.experiment_running = False
            self.socketio.emit('status', {'message': 'Experiment completed'})
        except Exception as e:
            error_tb = traceback.format_exc()
            print(f"DEBUG: Error running experiment: {e}")
            print(f"DEBUG: Traceback: {error_tb}")
            logger.error(f"Error running experiment: {e}")
            self.experiment_running = False  # Reset state
            self.socketio.emit('status', {'message': f'Error: {str(e)}'})
    
    def start(self, open_browser: bool = True):
        """
        Start the dashboard server in a separate thread.
        
        Args:
            open_browser: Whether to automatically open the dashboard in a browser
        """
        if self.running:
            logger.warning("Dashboard server already running")
            return
        
        def run_server():
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        self.running = True
        
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
        if not self.running:
            return
        
        # Shutdown the server
        self.socketio.stop()
        self.running = False
        logger.info("Dashboard server stopped")
    
    def send_reasoning_summary(self, problem_id, summary):
        """Send a reasoning summary for a problem."""
        self.socketio.emit('reasoning_summary', {
            'problem_id': problem_id,
            'summary': summary
        }) 