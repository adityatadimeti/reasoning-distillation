#!/bin/bash

# Script to test sequential processing with parallel_simple.csv for comparison

# Set up colors for better visualization
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}==== SEQUENTIAL PROCESSING TEST ====${NC}"
echo -e "${BLUE}Testing with 10 simple problems from parallel_simple.csv${NC}"
echo -e "${BLUE}Using sequential processing for comparison${NC}"
echo ""

# Record start time
start_time=$(date +%s)
echo -e "${YELLOW}Started at: $(date)${NC}"
echo "--------------------------------------------------------"

# Run the experiment with sequential processing (no --parallel flag)
# Arguments:
# - parallel_test: The config name (will look for config/experiments/parallel_test.yaml)
# - --verbose: Show more detailed logs including model calls

python run_experiment.py parallel_test --verbose

# Record end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "--------------------------------------------------------"
echo -e "${YELLOW}Completed at: $(date)${NC}"
echo -e "${GREEN}Total execution time: ${minutes}m ${seconds}s${NC}"
echo ""
echo -e "${BLUE}Results are saved in: ./results/parallel_test/${NC}"

# Add instructions
echo ""
echo -e "${GREEN}TIP: Compare the execution time with run_parallel_test.sh${NC}"
echo -e "${GREEN}     to see the performance improvement from parallelization.${NC}" 