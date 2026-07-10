#!/bin/bash

SESSION_NAME="stress-inference"
PYTHON_BIN="ekonomi-syariah-chatbot-llm/venv/bin/python"
SCRIPT_PATH="ekonomi-syariah-chatbot-llm/scripts/stress_test_inference.py"
LOG_FILE="stress_test_vram.log"

# Kill existing session if any
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new detached session
tmux new-session -d -s $SESSION_NAME

# Pane 1: Run Instance 1
tmux send-keys -t $SESSION_NAME "$PYTHON_BIN $SCRIPT_PATH" C-m

# Split window and Pane 2: Run Instance 2
tmux split-window -h -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME "$PYTHON_BIN $SCRIPT_PATH" C-m

# Pane 3: Log VRAM usage
tmux split-window -v -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME "while true; do date >> $LOG_FILE; nvidia-smi --query-gpu=memory.used,memory.free --format=csv >> $LOG_FILE; sleep 5; done" C-m

echo "🚀 Tmux session '$SESSION_NAME' started with 2 instances."
echo "📊 VRAM logs will be saved to: $LOG_FILE"
echo "🔗 To attach and see the progress, run: tmux attach -t $SESSION_NAME"
