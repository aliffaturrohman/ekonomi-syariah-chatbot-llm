# ekonomi-syariah-chatbot-llm

EKONOMI-SYARIAH-CHATBOT-LLM
│
├── .env                     # Kunci API (HuggingFace Token, OpenAI, dll)
├── .gitignore               # Abaikan /models, /vector_store, .env, data/training
├── README.md                # Dokumentasi: 1. Setup, 2. Run Ingest, 3. Run Build Dataset, 4. Run Finetune, 5. Run App
├── requirements.txt         # (Akan ada tambahan: transformers, peft, trl, accelerate, datasets)
│
├── config/
│   └── settings.py          # Nama base model, path model hasil fine-tune, K-value untuk retriever
│
├── data/
│   ├── raw/                 # Dokumen sumber (PDF, .txt, .md) yang jadi knowledge base
│   ├── qa_source/           # [RAFT] Kumpulan Pertanyaan & Jawaban "emas" (golden Q&A)
│   │   └── questions.jsonl  # Format: {"question": "...", "golden_answer": "...", "source_doc_id": "..."}
│   └── training/            # [RAFT] Dataset training yang *dihasilkan* dalam format RAFT
│       └── raft_train_dataset.jsonl
│
├── vector_store/            # Database vektor dari dokumen di /data/raw/
│
├── models/                  # [RAFT] Direktori untuk menyimpan model hasil fine-tuning
│   └── .gitkeep             # (Misal: my-raft-tuned-model/)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_processing.py   # Logika untuk Marker/Surya, chunking (dari /data/raw/)
│   ├── dataset_builder.py   # [RAFT] Logika untuk membuat dataset RAFT
│   ├── prompt_templates.py  # Template prompt untuk RAFT (training & inference)
│   │
│   ├── fine_tuning/         # [RAFT] Modul untuk proses fine-tuning
│   │   ├── __init__.py
│   │   └── train.py         # Skrip utama training (menggunakan SFTTrainer dari TRL)
│   │
│   └── inference/           # Modul untuk RAG (saat runtime)
│       ├── __init__.py
│       ├── chain.py         # Logika RAG chain (yang memuat model dari /models/)
│       └── retriever.py     # Logika untuk mengambil dokumen dari /vector_store/
│
├── scripts/                 # Skrip "runner" untuk pipeline
│   ├── 1_ingest_source_docs.py  # Menjalankan data_processing.py untuk mengisi vector_store
│   ├── 2_create_raft_dataset.py # [RAFT] Menjalankan dataset_builder.py
│   ├── 3_run_finetune.py        # [RAFT] Menjalankan src.fine_tuning.train.py
│
├── app/
│   └── main.py              # Aplikasi FastAPI/Streamlit (menggunakan src.inference.chain)
│
└── notebooks/
    ├── 01_data_processing_test.ipynb  # Eksperimen Marker/Surya/Chunking
    ├── 02_dataset_builder_test.ipynb  # [RAFT] Eksperimen membuat data training
    ├── 03_finetuning_test.ipynb       # [RAFT] Eksperimen training loop
    └── 04_inference_test.ipynb        # Menguji RAG chain dengan model yang sudah di-tune
