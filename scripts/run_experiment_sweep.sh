#!/bin/bash
# Script to run a sweep of experiments with different configurations

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

# Enable API calls for this run
export ENABLE_API_CALLS=1

# Default values
EXPERIMENT_TYPE="recursive"  # Options: baseline, summary, recursive
MAX_PROBLEMS=5  # Limit to 5 problems by default for quicker testing
SPLIT="test"
MAX_ITERATIONS=2
MODEL_LIST="deepseek-r1,deepseek-r1-distill-qwen-7b,deepseek-r1-distill-qwen-1p5b"
SUMMARIZATION_METHODS="self,external"

# Create output directories if they don't exist
mkdir -p results
mkdir -p logs
mkdir -p configs/temp

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --experiment-type)
            EXPERIMENT_TYPE="$2"
            shift
            shift
            ;;
        --max-problems)
            MAX_PROBLEMS="$2"
            shift
            shift
            ;;
        --split)
            SPLIT="$2"
            shift
            shift
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift
            shift
            ;;
        --models)
            MODEL_LIST="$2"
            shift
            shift
            ;;
        --summarization-methods)
            SUMMARIZATION_METHODS="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert comma-separated lists to arrays
IFS=',' read -ra MODELS <<< "$MODEL_LIST"
IFS=',' read -ra METHODS <<< "$SUMMARIZATION_METHODS"

# Print sweep configuration
echo "Running experiment sweep with the following configuration:"
echo "  Experiment type: $EXPERIMENT_TYPE"
echo "  Max problems: $MAX_PROBLEMS"
echo "  Split: $SPLIT"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Models: ${MODELS[*]}"
echo "  Summarization methods: ${METHODS[*]}"

# Timestamp for this sweep
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="results/sweep_${EXPERIMENT_TYPE}_${TIMESTAMP}"
mkdir -p "$SWEEP_DIR"

# Create a log file for the sweep
SWEEP_LOG="$SWEEP_DIR/sweep_log.txt"
echo "Experiment sweep started at $(date)" > "$SWEEP_LOG"
echo "Configuration:" >> "$SWEEP_LOG"
echo "  Experiment type: $EXPERIMENT_TYPE" >> "$SWEEP_LOG"
echo "  Max problems: $MAX_PROBLEMS" >> "$SWEEP_LOG"
echo "  Split: $SPLIT" >> "$SWEEP_LOG"
echo "  Max iterations: $MAX_ITERATIONS" >> "$SWEEP_LOG"
echo "  Models: ${MODELS[*]}" >> "$SWEEP_LOG"
echo "  Summarization methods: ${METHODS[*]}" >> "$SWEEP_LOG"
echo "---" >> "$SWEEP_LOG"

# Function to run a single experiment
run_experiment() {
    local model="$1"
    local method="$2"
    local output_dir="$SWEEP_DIR/${model}_${method}"
    mkdir -p "$output_dir"
    
    echo "Running experiment with model=$model, method=$method"
    echo "$(date): Running experiment with model=$model, method=$method" >> "$SWEEP_LOG"
    
    # Determine which script to use based on experiment type
    local script=""
    case "$EXPERIMENT_TYPE" in
        baseline)
            script="./scripts/run_baseline.sh"
            ;;
        summary)
            script="./scripts/run_summary_experiment.sh"
            ;;
        recursive)
            script="./scripts/run_recursive_pipeline.sh"
            ;;
        *)
            echo "Unknown experiment type: $EXPERIMENT_TYPE"
            return 1
            ;;
    esac
    
    # Build command based on experiment type
    local cmd="$script --model $model --max-problems $MAX_PROBLEMS --split $SPLIT"
    
    if [ "$EXPERIMENT_TYPE" != "baseline" ]; then
        cmd="$cmd --max-iterations $MAX_ITERATIONS"
        
        if [ "$EXPERIMENT_TYPE" = "summary" ]; then
            cmd="$cmd --method $method"
        fi
    fi
    
    # Record the command
    echo "Command: $cmd" | tee -a "$SWEEP_LOG"
    
    # Run the experiment and capture output
    start_time=$(date +%s)
    $cmd > "$output_dir/output.log" 2>&1
    status=$?
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    
    # Record the result
    if [ $status -eq 0 ]; then
        echo "Experiment completed successfully in $runtime seconds" | tee -a "$SWEEP_LOG"
        
        # Copy results to the sweep directory
        find results -name "*.json" -newer "$output_dir/output.log" -exec cp {} "$output_dir/" \;
        
        # Extract and record metrics (customize based on your output format)
        if grep -q "accuracy:" "$output_dir/output.log"; then
            accuracy=$(grep "accuracy:" "$output_dir/output.log" | awk '{print $2}')
            echo "Accuracy: $accuracy" | tee -a "$SWEEP_LOG" "$output_dir/metrics.txt"
        fi
        
        if grep -q "improvement_rate:" "$output_dir/output.log"; then
            improvement=$(grep "improvement_rate:" "$output_dir/output.log" | awk '{print $2}')
            echo "Improvement rate: $improvement" | tee -a "$SWEEP_LOG" "$output_dir/metrics.txt"
        fi
    else
        echo "Experiment failed with status $status after $runtime seconds" | tee -a "$SWEEP_LOG"
    fi
    
    echo "---" >> "$SWEEP_LOG"
}

# Run experiments for all combinations
for model in "${MODELS[@]}"; do
    if [ "$EXPERIMENT_TYPE" = "baseline" ]; then
        # Baseline doesn't use summarization methods
        run_experiment "$model" "none"
    else
        # Summary and recursive use summarization methods
        for method in "${METHODS[@]}"; do
            run_experiment "$model" "$method"
        done
    fi
done

echo "Experiment sweep completed at $(date)" | tee -a "$SWEEP_LOG"
echo "Results saved to $SWEEP_DIR"

# Generate a summary of all experiments
echo "Generating summary..."

# Create summary CSV
SUMMARY_CSV="$SWEEP_DIR/summary.csv"
echo "model,method,accuracy,improvement_rate,runtime" > "$SUMMARY_CSV"

# Extract metrics from all experiments
for model in "${MODELS[@]}"; do
    if [ "$EXPERIMENT_TYPE" = "baseline" ]; then
        methods=("none")
    else
        methods=("${METHODS[@]}")
    fi
    
    for method in "${methods[@]}"; do
        output_dir="$SWEEP_DIR/${model}_${method}"
        if [ -f "$output_dir/metrics.txt" ]; then
            accuracy=$(grep "Accuracy:" "$output_dir/metrics.txt" | awk '{print $2}' || echo "N/A")
            improvement=$(grep "Improvement rate:" "$output_dir/metrics.txt" | awk '{print $2}' || echo "N/A")
            runtime=$(grep "Experiment completed successfully in" "$output_dir/output.log" | awk '{print $5}' || echo "N/A")
            
            echo "$model,$method,$accuracy,$improvement,$runtime" >> "$SUMMARY_CSV"
        else
            echo "$model,$method,FAILED,FAILED,FAILED" >> "$SUMMARY_CSV"
        fi
    done
done

echo "Summary saved to $SUMMARY_CSV"