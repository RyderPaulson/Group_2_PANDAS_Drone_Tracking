#!/bin/bash

# Drone Tracking System Launch Script
# Usage: ./launch.sh [options]

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
CAMERA_ID=0
TEXT_PROMPT="drone"
TEST_NAME="default"
WINDOW_SIZE=16
RST_INTERVAL_MULT=16
BB_CHECK_MULT=8

# Flags
SEND_TO_BOARD=false
PRINT_COORD=false
WRITE_OUT=false
DISP_OUT=false
BENCHMARKING=false

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Drone tracking system using CoTracker and Grounding DINO

OPTIONS:
    -c, --camera <id>           Camera ID (integer) or path to video file (default: media/ds_pan_cut.mp4)
    -t, --text-prompt <text>    Text prompt for detection (default: "drone")
    -n, --test-name <name>      Name for test run (default: "default")
    -w, --window-size <size>    CoTracker window size (default: 16)
    -r, --rst-mult <mult>       Reset interval multiplier (default: 16)
    -b, --bb-mult <mult>        Bounding box check multiplier (default: 8)

    --send-to-board             Send coordinates to board
    --print-coord               Print coordinates to console
    --write-out                 Write output to file
    --disp-out                  Display output window
    --benchmarking              Enable benchmarking mode

    -h, --help                  Display this help message

EXAMPLES:
    # Run with video file and display output
    $0 -c media/video.mp4 --disp-out --print-coord

    # Run with camera 0, benchmarking enabled
    $0 -c 0 --benchmarking --write-out

    # Track a specific object with custom settings
    $0 -c media/bird.mp4 -t "bird" -n "bird_test" --disp-out

EOF
    exit 0
}

# Print message with color
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--camera)
            CAMERA_ID="$2"
            shift 2
            ;;
        -t|--text-prompt)
            TEXT_PROMPT="$2"
            shift 2
            ;;
        -n|--test-name)
            TEST_NAME="$2"
            shift 2
            ;;
        -w|--window-size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        -r|--rst-mult)
            RST_INTERVAL_MULT="$2"
            shift 2
            ;;
        -b|--bb-mult)
            BB_CHECK_MULT="$2"
            shift 2
            ;;
        --send-to-board)
            SEND_TO_BOARD=true
            shift
            ;;
        --print-coord)
            PRINT_COORD=true
            shift
            ;;
        --write-out)
            WRITE_OUT=true
            shift
            ;;
        --disp-out)
            DISP_OUT=true
            shift
            ;;
        --benchmarking)
            BENCHMARKING=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed or not in PATH"
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    print_error "main.py not found in current directory"
    exit 1
fi

# Build the command
CMD="python3 main.py $CAMERA_ID"
CMD="$CMD --text-prompt \"$TEXT_PROMPT\""
CMD="$CMD --test-name \"$TEST_NAME\""
CMD="$CMD --window-size $WINDOW_SIZE"
CMD="$CMD --rst-interval-mult $RST_INTERVAL_MULT"
CMD="$CMD --bb-check-mult $BB_CHECK_MULT"

[ "$SEND_TO_BOARD" = true ] && CMD="$CMD --send-to-board"
[ "$PRINT_COORD" = true ] && CMD="$CMD --print-coord"
[ "$WRITE_OUT" = true ] && CMD="$CMD --write-out"
[ "$DISP_OUT" = true ] && CMD="$CMD --disp-out"
[ "$BENCHMARKING" = true ] && CMD="$CMD --benchmarking"

# Print configuration
print_info "Starting drone tracking system..."
print_info "Camera: $CAMERA_ID"
print_info "Text prompt: $TEXT_PROMPT"
print_info "Test name: $TEST_NAME"
print_info "Window size: $WINDOW_SIZE"

# Execute the command
print_info "Executing: $CMD"
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    print_info "Program completed successfully"
else
    print_error "Program exited with error code $?"
    exit 1
fi