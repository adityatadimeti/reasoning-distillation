"""
Dashboard server for monitoring recursive pipeline experiments.
"""
from flask import Flask, render_template, jsonify, Response, send_from_directory
import json
import time
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Default path for dashboard data
DEFAULT_DATA_PATH = Path("dashboard/experiment_data.json")

def get_data_path():
    """Get the path to the dashboard data file."""
    env_path = os.environ.get("DASHBOARD_DATA_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DATA_PATH

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/data')
@app.route('/api/data')
def get_data():
    """API endpoint to get the current experiment data."""
    data_path = get_data_path()
    if data_path.exists():
        try:
            with open(data_path, 'r') as f:
                return jsonify(json.load(f))
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {data_path}")
            return jsonify({"status": "error", "message": "Invalid JSON data"})
        except Exception as e:
            logger.error(f"Error reading data file: {str(e)}")
            return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "no_data", "message": "No experiment data available"})

@app.route('/events')
@app.route('/api/events')
def events():
    """Server-Sent Events endpoint for real-time updates."""
    def generate():
        # Send initial retry directive to control reconnection frequency
        yield "retry: 5000\n\n"  # Set retry interval to 5 seconds
        
        last_modified = 0
        data_path = get_data_path()
        
        while True:
            if data_path.exists():
                try:
                    current_modified = os.path.getmtime(data_path)
                    if current_modified > last_modified:
                        last_modified = current_modified
                        with open(data_path, 'r') as f:
                            data = json.load(f)
                        yield f"data: {json.dumps(data)}\n\n"
                except Exception as e:
                    logger.error(f"Error in SSE generation: {str(e)}")
                    yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
            time.sleep(1)
    
    response = Response(generate(), mimetype='text/event-stream')
    # Add headers to prevent caching
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # For Nginx
    return response

@app.route('/api/updates')
def updates():
    """Dummy endpoint for compatibility."""
    response = jsonify({"status": "ok"})
    # Add cache control header to reduce polling frequency
    response.headers['Cache-Control'] = 'public, max-age=5'  # Cache for 5 seconds
    return response

@app.route('/api/status')
def status():
    """Dummy endpoint for compatibility."""
    response = jsonify({"status": "ok"})
    # Add cache control header to reduce polling frequency
    response.headers['Cache-Control'] = 'public, max-age=5'  # Cache for 5 seconds
    return response

@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

def initialize_data_file():
    """Initialize the dashboard data file if it doesn't exist."""
    data_path = get_data_path()
    if not data_path.exists():
        os.makedirs(data_path.parent, exist_ok=True)
        initial_data = {
            "status": "waiting",
            "problems": {},
            "current_problem": None,
            "api_calls": [],
            "start_time": time.time(),
            "last_update": time.time()
        }
        with open(data_path, 'w') as f:
            json.dump(initial_data, f, indent=2)
        logger.info(f"Initialized dashboard data file at {data_path}")

if __name__ == '__main__':
    # Initialize data file
    initialize_data_file()
    
    # Get port from environment or use default
    port = int(os.environ.get("DASHBOARD_PORT", 5000))
    
    logger.info(f"Starting dashboard server on port {port}")
    logger.info(f"Dashboard available at http://localhost:{port}")
    
    # Start the Flask app
    app.run(debug=True, port=port, threaded=True) 