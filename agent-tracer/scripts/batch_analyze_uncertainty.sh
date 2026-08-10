#!/bin/bash
#
# Batch Uncertainty Analysis Script
#
# Analyzes all simulation files in data/simulations/ and saves results to data/uncertainty/
# with matching filenames.
#
# Usage:
#   ./scripts/batch_analyze_uncertainty.sh
#   ./scripts/batch_analyze_uncertainty.sh --verbose
#   ./scripts/batch_analyze_uncertainty.sh --detailed

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
VERBOSE_FLAG=""
DETAILED_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE_FLAG="--verbose"
            shift
            ;;
        --detailed|-d)
            DETAILED_FLAG="--detailed"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --verbose, -v    Include verbose statistics"
            echo "  --detailed, -d   Show detailed turn-by-turn view"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --verbose"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Directories
SIMULATIONS_DIR="$PROJECT_ROOT/data/simulations"
UNCERTAINTY_DIR="$PROJECT_ROOT/data/uncertainty"

# Check if simulations directory exists
if [ ! -d "$SIMULATIONS_DIR" ]; then
    echo -e "${RED}❌ Error: Simulations directory not found: $SIMULATIONS_DIR${NC}"
    exit 1
fi

# Count simulation files
TOTAL_FILES=$(find "$SIMULATIONS_DIR" -name "*.json" -type f | wc -l | tr -d ' ')

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No simulation files found in $SIMULATIONS_DIR${NC}"
    exit 0
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Batch Uncertainty Analysis${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Found $TOTAL_FILES simulation file(s)${NC}"
echo -e "${GREEN}Output directory: $UNCERTAINTY_DIR${NC}"
echo ""

# Initialize counters
PROCESSED=0
SKIPPED=0
FAILED=0

# Process each simulation file
for sim_file in "$SIMULATIONS_DIR"/*.json; do
    # Skip if no files found (glob doesn't match)
    [ -e "$sim_file" ] || continue
    
    filename=$(basename "$sim_file")
    uncertainty_file="$UNCERTAINTY_DIR/$filename"
    
    # Check if analysis already exists
    if [ -f "$uncertainty_file" ]; then
        echo -e "${YELLOW}⏭️  Skipping $filename (already analyzed)${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    echo -e "${BLUE}📊 Analyzing: $filename${NC}"
    
    # Run analysis
    if python -m tau2.scripts.analyze_uncertainty "$sim_file" $VERBOSE_FLAG $DETAILED_FLAG 2>&1 | grep -q "Analysis complete"; then
        echo -e "${GREEN}✅ Completed: $filename${NC}"
        PROCESSED=$((PROCESSED + 1))
    else
        echo -e "${RED}❌ Failed: $filename${NC}"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Processed: $PROCESSED${NC}"
echo -e "${YELLOW}⏭️  Skipped: $SKIPPED (already analyzed)${NC}"
if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}❌ Failed: $FAILED${NC}"
fi
echo -e "${BLUE}========================================${NC}"

# Exit with appropriate code
if [ "$FAILED" -gt 0 ]; then
    exit 1
else
    echo -e "${GREEN}🎉 Batch analysis complete!${NC}"
    exit 0
fi

