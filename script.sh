#!/bin/bash

echo "🚀 Mulai membuat struktur proyek RAFT..."

# 1. Buat semua direktori
echo "Membuat direktori..."
mkdir -p config
mkdir -p data/raw
mkdir -p data/qa_source
mkdir -p data/training
mkdir -p vector_store
mkdir -p models
mkdir -p src/fine_tuning
mkdir -p src/inference
mkdir -p scripts
mkdir -p app
mkdir -p notebooks

# 2. Buat semua file kosong
echo "Membuat file-file..."

# File di root
touch .env
touch .gitignore
touch README.md
touch requirements.txt

# Folder config
touch config/settings.py

# Folder data
touch data/qa_source/questions.jsonl
touch data/training/raft_train_dataset.jsonl
# (Kita biarkan /data/raw/ kosong untuk Anda isi)

# Placeholder untuk folder kosong
touch vector_store/.gitkeep
touch models/.gitkeep

# Folder src
touch src/__init__.py
touch src/data_processing.py
touch src/dataset_builder.py
touch src/prompt_templates.py

touch src/fine_tuning/__init__.py
touch src/fine_tuning/train.py

touch src/inference/__init__.py
touch src/inference/chain.py
touch src/inference/retriever.py

# Folder scripts
touch scripts/1_ingest_source_docs.py
touch scripts/2_create_raft_dataset.py
touch scripts/3_run_finetune.py

# Folder app
touch app/main.py

# Folder notebooks
touch notebooks/01_data_processing_test.ipynb
touch notebooks/02_dataset_builder_test.ipynb
touch notebooks/03_finetuning_test.ipynb
touch notebooks/04_inference_test.ipynb

echo "✅ Selesai! Struktur proyek telah dibuat."
echo "Jangan lupa untuk mengisi .env, requirements.txt, dan data/raw/."
