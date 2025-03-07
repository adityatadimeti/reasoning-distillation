#!/bin/bash
# Script to start the dashboard server

# Set the Python path to include the project root
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Default port
PORT=5000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --port)
            PORT="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set environment variables
export DASHBOARD_PORT=$PORT

# Check if Flask is installed
if ! python -c "import flask" &> /dev/null; then
    echo "Flask is not installed. Installing..."
    pip install flask
fi

echo "Starting dashboard server on port $PORT..."
echo "Dashboard will be available at http://localhost:$PORT"

# Start the dashboard server
python dashboard/app.py 