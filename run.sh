#!/bin/bash
# Unified Automation Script for training and evaluation

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment (venv) not found at $SCRIPT_DIR/venv"
    echo "👉 Please create the virtual environment and install dependencies first."
    exit 1
fi

PYTHON_EXEC="./venv/bin/python"

# Parse arguments
ACTION=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --run)
            ACTION="$2"
            shift 2
            ;;
        training|evaluate)
            ACTION="$1"
            shift
            ;;
        *)
            echo "❌ Error: Unknown argument $1"
            echo "Usage: $0 --run [training|evaluate]  OR  $0 [training|evaluate]"
            exit 1
            ;;
    esac
done

if [ "$ACTION" == "training" ]; then
    echo "✅ Running training queue..."
    $PYTHON_EXEC scripts/04_train_model.py --queue
elif [ "$ACTION" == "evaluate" ]; then
    echo "✅ Running evaluation check..."
    # By default, use direct DeepSeek API for evaluation as configured in .env
    $PYTHON_EXEC scripts/07_run_ragas_evaluation.py --judge deepseek
else
    echo "❌ Error: Missing or invalid action."
    echo "Usage: $0 --run [training|evaluate]  OR  $0 [training|evaluate]"
    exit 1
fi
