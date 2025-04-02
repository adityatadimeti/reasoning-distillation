#!/bin/bash
# Script to run GPQA experiments in batches of 8 problems
# and merge results after each batch

# Base configuration
EXPERIMENT="gpqa_diamond"  # Experiment name
BATCH_SIZE=8               # Number of problems per batch
START_INDEX=16             # Starting problem index (we've already done 0-15)
END_INDEX=197              # Total number of problems in dataset (adjust if needed)
MERGED_FILE="results/summarization_8_iter_rzn-R1_summ-V3_gpqa_diamond_mc/merged_results_all.json"
INITIAL_MERGED="results/summarization_8_iter_rzn-R1_summ-V3_gpqa_diamond_mc/merged_results_0_to_15.json"

# Create a temporary directory for batch result files
mkdir -p results/tmp_batches

# Copy the initial merged file (0-15) to use as our base
cp "$INITIAL_MERGED" "$MERGED_FILE"
echo "Starting with base merged file: $MERGED_FILE"

# Function to run a batch and update the merged results
run_batch() {
    local start=$1
    local end=$2
    local batch_num=$(( (start - START_INDEX) / BATCH_SIZE + 1 ))
    
    echo "==============================================="
    echo "Running batch $batch_num: Problems $start-$end"
    echo "==============================================="
    
    # Run the experiment
    python run_experiment.py "$EXPERIMENT" --index-range "$start-$end" --parallel --concurrency 8 --verbose
    
    # Wait for a moment to make sure files are saved
    sleep 2
    
    # Find the most recent result directory for this experiment
    latest_dir=$(find results/summarization_8_iter_rzn-R1_summ-V3_gpqa_diamond_mc -maxdepth 1 -type d | sort -r | head -1)
    latest_results="$latest_dir/results.json"
    
    echo "Latest results found: $latest_results"
    
    # Merge with existing results
    echo "Merging batch $batch_num with previous results..."
    python merge_experiment_results.py --input "$MERGED_FILE" "$latest_results" --output "$MERGED_FILE"
    
    # Print a quick summary of correct answers in this batch
    echo "Quick summary of batch $batch_num:"
    grep -A 1 '"correct":' "$latest_results" | grep true | wc -l
    
    echo "Batch $batch_num complete"
    echo ""
}

# Calculate how many full batches we need
FULL_BATCHES=$(( (END_INDEX - START_INDEX + 1) / BATCH_SIZE ))
REMAINDER=$(( (END_INDEX - START_INDEX + 1) % BATCH_SIZE ))

echo "Will run $FULL_BATCHES full batches of $BATCH_SIZE problems"
if [ $REMAINDER -gt 0 ]; then
    echo "Plus 1 partial batch of $REMAINDER problems"
fi

# Run full batches
for ((i=0; i<FULL_BATCHES; i++)); do
    batch_start=$((START_INDEX + i*BATCH_SIZE))
    batch_end=$((batch_start + BATCH_SIZE - 1))
    run_batch $batch_start $batch_end
done

# Run the remainder batch if needed
if [ $REMAINDER -gt 0 ]; then
    batch_start=$((START_INDEX + FULL_BATCHES*BATCH_SIZE))
    batch_end=$END_INDEX
    run_batch $batch_start $batch_end
fi

echo "All batches complete!"
echo "Final merged results are in: $MERGED_FILE"

# Print summary of all results
echo "Running final analysis on all results..."
python show_answer_progression.py "$MERGED_FILE" | grep "Final answer correct: Yes" | wc -l 