#!/bin/bash

# Script to test parallel processing with parallel_simple.csv

# Set up colors for better visualization
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}==== PARALLEL PROCESSING TEST ====${NC}"
echo -e "${BLUE}Testing with 10 simple problems from parallel_simple.csv${NC}"
echo -e "${BLUE}Using concurrency level of 4 (processing 4 problems simultaneously)${NC}"
echo ""

# Record start time
start_time=$(date +%s)
echo -e "${YELLOW}Started at: $(date)${NC}"
echo "--------------------------------------------------------"

# Run the experiment with parallel processing enabled
# Arguments:
# - parallel_test: The config name (will look for config/experiments/parallel_test.yaml)
# - --parallel: Enable parallel processing
# - --concurrency: Number of problems to process in parallel
# - --verbose: Show more detailed logs including model calls

python run_experiment.py parallel_test --parallel --concurrency 4 --verbose

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

# Add instructions for examining the results
echo ""
echo -e "${GREEN}TIP: To verify parallel execution, examine the logs for timestamps${NC}"
echo -e "${GREEN}     showing multiple problems being processed simultaneously.${NC}" 