#!/bin/bash
# Auto-pipeline: after DeepSeek inference completes → BLEU/ROUGE → RAGAS → finding.md
set -e

cd "$(dirname "$0")"
VENV="$(pwd)/venv/bin/python"
SCRIPT_DIR="$(pwd)/scripts"
LOG="logs/pipeline.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait for all 3 deepseek result files
log "⏳ Menunggu DeepSeek inference selesai..."
while true; do
    s1_ok=0; s2_ok=0; s3_ok=0
    for f in eval_results_full/results_deepseek_v4_flash_strat1.json; do
        [ -f "$f" ] && [ $(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null) -eq 291 ] && s1_ok=1
    done
    for f in eval_results_full/results_deepseek_v4_flash_strat2.json; do
        [ -f "$f" ] && [ $(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null) -eq 291 ] && s2_ok=1
    done
    for f in eval_results_full/results_deepseek_v4_flash_strat3.json; do
        [ -f "$f" ] && [ $(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null) -eq 291 ] && s3_ok=1
    done
    [ $s1_ok -eq 1 ] && [ $s2_ok -eq 1 ] && [ $s3_ok -eq 1 ] && break
    sleep 60
done
log "✅ Semua DeepSeek inference selesai!"

# Step 1: BLEU/ROUGE
log "📊 Menghitung BLEU & ROUGE untuk semua model..."
$VENV "$SCRIPT_DIR/finalize_evaluation_all.py" 2>&1 | tee -a "$LOG"

# Step 2: RAGAS
log "🤖 Menjalankan RAGAS evaluation dengan juri DeepSeek..."
$VENV "$SCRIPT_DIR/07_run_ragas_evaluation.py" --judge deepseek 2>&1 | tee -a "$LOG"

log "🏁 Pipeline selesai! Hasil di eval_metrics/ dan bisa ditulis ke finding.md."
