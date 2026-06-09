#!/bin/bash

SESSION_NAME="hanif_gen"

# Cek apakah sesi sudah ada
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  echo "🚀 Memulai sesi tmux baru: $SESSION_NAME"
  
  # Buat sesi baru dan jalankan pipeline
  tmux new-session -d -t $SESSION_NAME
  
  # Jalankan perintah di dalam tmux
  # 1. Masuk venv
  # 2. Hapus data lama (opsional, agar bersih)
  # 3. Jalankan generator
  # 4. Jalankan pembersih noise setelah generator selesai
  tmux send-keys -t $SESSION_NAME "source venv/bin/activate" C-m
  tmux send-keys -t $SESSION_NAME "rm -rf data/dataset_training_ver2/*" C-m
  tmux send-keys -t $SESSION_NAME "python scripts/03_generate_dataset.py && python scripts/fix_dataset_noise.py" C-m
  
  echo "✅ Generator berjalan di background (tmux)."
  echo "👉 Gunakan: 'tmux attach -t $SESSION_NAME' untuk melihat proses."
  echo "👉 Gunakan: 'Ctrl+B lalu D' untuk keluar dari tampilan tmux tanpa mematikan proses."
else
  echo "⚠️ Sesi $SESSION_NAME sudah berjalan."
  echo "👉 Gunakan: 'tmux attach -t $SESSION_NAME' untuk melihat."
fi
