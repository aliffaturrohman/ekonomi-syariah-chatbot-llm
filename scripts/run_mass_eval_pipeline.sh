#!/bin/bash

SESSION_NAME="mass-eval"
PYTHON_BIN="ekonomi-syariah-chatbot-llm/venv/bin/python"
INF_SCRIPT="ekonomi-syariah-chatbot-llm/scripts/06_mass_inference_turbo.py"
RAGAS_SCRIPT="ekonomi-syariah-chatbot-llm/scripts/07_run_ragas_evaluation.py"
LOG_FILE="mass_eval_pipeline.log"

# Kill existing session if any
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new detached session
tmux new-session -d -s $SESSION_NAME -n "Pipeline"

# Pane 1: Run Inference then Ragas
tmux send-keys -t $SESSION_NAME "echo '--- START PIPELINE ---' > $LOG_FILE" C-m
tmux send-keys -t $SESSION_NAME "$PYTHON_BIN $INF_SCRIPT 2>&1 | tee -a $LOG_FILE && $PYTHON_BIN $RAGAS_SCRIPT --judge deepseek 2>&1 | tee -a $LOG_FILE" C-m

# Pane 2: VRAM Monitor
tmux split-window -h -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME "watch -n 5 nvidia-smi" C-m

echo "🚀 Tmux session '$SESSION_NAME' started."
echo "📝 Logs: $LOG_FILE"
echo "🔗 Monitor progress: tmux attach -t $SESSION_NAME"
