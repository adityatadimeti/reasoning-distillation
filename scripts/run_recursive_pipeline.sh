#!/bin/bash
# Script to run the recursive reasoning refinement experiment

# Set the Python path to include the project root
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found"
fi

# Verify API key is set
if [ -z "$FIREWORKS_API_KEY" ]; then
    echo "Error: FIREWORKS_API_KEY environment variable not set"
    echo "Please create a .env file with your API key or set it in your environment"
    exit 1
fi

# Default values
CONFIG=""
USE_SUMMARIZE_TAGS=false
SPLIT="test"
MAX_PROBLEMS=5  # Limit to 5 problems by default for quicker testing
MAX_ITERATIONS=3  # Default to 3 iterations
MODEL=""
DATASET=""
ENABLE_DASHBOARD=false
DASHBOARD_PORT=5000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            CONFIG="$2"
            shift
            shift
            ;;
        --use-summarize-tags)
            USE_SUMMARIZE_TAGS=true
            shift
            ;;
        --split)
            SPLIT="$2"
            shift
            shift
            ;;
        --max-problems)
            MAX_PROBLEMS="$2"
            shift
            shift
            ;;
        --problem-ids)
            PROBLEM_IDS="$2"
            shift
            shift
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift
            shift
            ;;
        --model)
            MODEL="$2"
            shift
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift
            shift
            ;;
        --all)
            MAX_PROBLEMS=""  # Process all problems
            shift
            ;;
        --dashboard)
            ENABLE_DASHBOARD=true
            shift
            ;;
        --dashboard-port)
            DASHBOARD_PORT="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default config based on tag option if not specified
if [ -z "$CONFIG" ]; then
    if [ "$USE_SUMMARIZE_TAGS" = true ]; then
        CONFIG="configs/experiments/summarize_tags.yaml"
    else
        CONFIG="configs/experiments/recursive_refinement.yaml"
    fi
fi

echo "Running recursive experiment with configuration:"
echo "  Config file: $CONFIG"
if [ "$USE_SUMMARIZE_TAGS" = true ]; then
    echo "  Using experimental summarize tags"
fi
echo "  Data split: $SPLIT"
echo "  Max iterations: $MAX_ITERATIONS"
if [ -n "$MAX_PROBLEMS" ]; then
    echo "  Max problems: $MAX_PROBLEMS"
else
    echo "  Processing all problems"
fi
if [ -n "$PROBLEM_IDS" ]; then
    echo "  Problem IDs: $PROBLEM_IDS"
fi
if [ -n "$MODEL" ]; then
    echo "  Model: $MODEL"
fi
if [ -n "$DATASET" ]; then
    echo "  Dataset: $DATASET"
fi
if [ "$ENABLE_DASHBOARD" = true ]; then
    echo "  Dashboard enabled on port $DASHBOARD_PORT"
fi

# Create output directories if they don't exist
mkdir -p results
mkdir -p logs
mkdir -p configs/temp
mkdir -p dashboard

# Enable API calls for this run
export ENABLE_API_CALLS=1

# Set up dashboard if enabled
if [ "$ENABLE_DASHBOARD" = true ]; then
    # Set environment variables for dashboard
    export ENABLE_EXPERIMENT_DASHBOARD=1
    export DASHBOARD_DATA_PATH="dashboard/experiment_data.json"
    export DASHBOARD_PORT=$DASHBOARD_PORT
    
    # Initialize dashboard data file
    echo '{
        "status": "initializing",
        "problems": {},
        "current_problem": null,
        "api_calls": [],
        "start_time": '$(date +%s)',
        "last_update": '$(date +%s)'
    }' > $DASHBOARD_DATA_PATH
    
    # Start dashboard server in background
    echo "Starting dashboard server on port $DASHBOARD_PORT"
    ./dashboard/start_dashboard.sh --port $DASHBOARD_PORT &
    DASHBOARD_PID=$!
    
    # Give the server a moment to start
    sleep 2
    
    # Open dashboard in browser if on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:$DASHBOARD_PORT"
    else
        echo "Dashboard available at: http://localhost:$DASHBOARD_PORT"
    fi
else
    # Disable dashboard
    export ENABLE_EXPERIMENT_DASHBOARD=0
fi

# Build command
CMD="python experiments/recursive_pipeline.py"
CMD="$CMD --config $CONFIG"
if [ "$USE_SUMMARIZE_TAGS" = true ]; then
    CMD="$CMD --use-summarize-tags"
fi
CMD="$CMD --split $SPLIT"
CMD="$CMD --max-iterations $MAX_ITERATIONS"
if [ -n "$MAX_PROBLEMS" ]; then
    CMD="$CMD --max-problems $MAX_PROBLEMS"
fi
if [ -n "$PROBLEM_IDS" ]; then
    CMD="$CMD --problem-ids $PROBLEM_IDS"
fi
if [ -n "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi
if [ -n "$DATASET" ]; then
    CMD="$CMD --dataset $DATASET"
fi

# Run the experiment
echo "Running command: $CMD"
eval $CMD
EXPERIMENT_STATUS=$?

# Check if the experiment was successful
if [ $EXPERIMENT_STATUS -eq 0 ]; then
    echo "Recursive experiment completed successfully"
else
    echo "Recursive experiment failed"
fi

# Update dashboard status if enabled
if [ "$ENABLE_DASHBOARD" = true ]; then
    if [ $EXPERIMENT_STATUS -eq 0 ]; then
        # Update status to completed
        jq '.status = "completed"' $DASHBOARD_DATA_PATH > temp.json && mv temp.json $DASHBOARD_DATA_PATH
    else
        # Update status to error
        jq '.status = "error"' $DASHBOARD_DATA_PATH > temp.json && mv temp.json $DASHBOARD_DATA_PATH
    fi
    
    # Keep dashboard running for a while to allow viewing results
    echo "Dashboard will remain active for 5 minutes. Press Ctrl+C to stop it earlier."
    echo "Dashboard URL: http://localhost:$DASHBOARD_PORT"
    sleep 300
    
    # Kill dashboard server
    if [ -n "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null || true
    fi
fi

exit $EXPERIMENT_STATUS