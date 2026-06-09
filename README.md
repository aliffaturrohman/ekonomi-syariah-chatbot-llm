# 🕌 HANIF: Asisten Ekonomi Syariah (Helpful AI for Noble Islamic Finance)

HANIF adalah sistem asisten AI pakar Ekonomi Syariah yang dibangun menggunakan teknik **RAFT (Retrieval-Augmented Fine-Tuning)**. Proyek ini menggabungkan kekuatan **RAG (Retrieval-Augmented Generation)** dengan **Fine-Tuning** model bahasa (LLM) untuk memberikan jawaban yang akurat, berbasis literatur, dan memiliki alur penalaran (**Chain-of-Thought**) yang transparan.

---

## 🌟 Fitur Utama

- **Pipeline Data Canggih:** Konversi PDF ke Markdown dengan dukungan OCR (`surya-ocr`) dan pembersihan teks otomatis menggunakan LLM.
- **Pakar Ekonomi Syariah (RAFT):** Model tidak hanya dilatih untuk menjawab, tapi juga dilatih untuk membedakan antara informasi yang relevan (*Oracle*) dan tidak relevan (*Distractors*) dalam konteks RAG.
- **Chain-of-Thought (CoT):** Output model mencakup tag `<thought>` yang berisi proses analisis sebelum memberikan jawaban akhir.
- **Efisiensi Tinggi:** Fine-tuning menggunakan **Unsloth** (LoRA/QLoRA) yang memungkinkan pelatihan model 7B pada GPU konsumen (misal: RTX 3060).
- **Multi-Antarmuka:** Tersedia dalam bentuk Web App (Streamlit), API (FastAPI), dan CLI.
- **Sinkronisasi Cloud:** Integrasi dengan Google Cloud Firestore untuk manajemen riwayat percakapan.

---

## 🏗️ Arsitektur Sistem

Proyek ini mengikuti alur kerja end-to-end mulai dari pengolahan data mentah hingga inferensi:

1.  **Ingestion:** PDF -> Markdown -> Cleaning -> Vector Store (ChromaDB).
2.  **Dataset Generation:** Pembuatan dataset RAFT menggunakan model *Teacher* (Ollama).
3.  **Training:** Fine-tuning base model (Qwen 2.5 7B) menggunakan dataset RAFT.
4.  **Export:** Konversi model ke format GGUF untuk deployment lokal.
5.  **Inference:** RAG + CoT via Streamlit/FastAPI/CLI.

---

## 🛠️ Teknologi yang Digunakan

| Kategori | Teknologi |
| :--- | :--- |
| **Base LLM** | Qwen 2.5 (7B / 14B) |
| **Orchestration** | LangChain, LangGraph |
| **Vector Database** | ChromaDB |
| **Embeddings** | `intfloat/multilingual-e5-large` |
| **Fine-Tuning** | Unsloth, LoRA, QLoRA, TRL |
| **Data Processing** | Marker, Surya-OCR, PyPDF |
| **Backend / UI** | FastAPI, Streamlit |
| **Cloud DB** | Google Cloud Firestore |
| **Inference Engine** | Ollama |

---

## 📁 Struktur Direktori

```text
ekonomi-syariah-chatbot-llm/
├── app/                # Aplikasi Inferensi (API, Web, CLI, Logic)
│   ├── app.py          # Dashboard Streamlit (Frontend Utama)
│   ├── app_cli.py      # Antarmuka Command Line
│   ├── main.py         # Backend FastAPI
│   ├── graph.py        # State Machine Alur Chat (LangGraph)
│   └── db.py           # Integrasi Firestore
├── config/             # Konfigurasi sistem
├── data/               # Penyimpanan Data
│   ├── raw/            # PDF Mentah
│   ├── processed/      # Hasil konversi Markdown & Cleaning
│   └── dataset_training/ # Dataset RAFT hasil generate
├── scripts/            # Script Pipeline (01-05)
│   ├── 01_pdf_to_md.py # PDF to Markdown
│   ├── 01b_cleaning.py # LLM Text Cleaning
│   ├── 02_ingest.py    # Ingest ke ChromaDB
│   ├── 03_generate.py  # RAFT Dataset Generator
│   ├── 04_train.py     # Unsloth Fine-Tuning
│   └── 05_export.py    # GGUF Export
├── models/             # LoRA Adapters & GGUF Models
├── notebooks/          # Eksperimen & Testing (Jupyter)
├── vector_store/       # Database Vektor (ChromaDB)
└── requirements.txt    # Daftar dependensi Python
```

---

## 🚀 Panduan Penggunaan

### 1. Instalasi & Persiapan
Pastikan Anda memiliki Python 3.10+ dan CUDA yang terkonfigurasi.

```bash
# Clone repository
git clone https://github.com/username/ekonomi-syariah-chatbot-llm.git
cd ekonomi-syariah-chatbot-llm

# Install dependencies
pip install -r requirements.txt

# Install Ollama (untuk inferensi lokal)
# Kunjungi https://ollama.com
```

### 2. ⚡ Otomatisasi Training & Evaluasi (Unified Runner)
Kami telah menyediakan script otomatisasi tunggal untuk mempermudah eksekusi training dan evaluasi model secara berantai.

#### Cara Penggunaan (Linux/macOS):
```bash
# Jalankan antrean training model
./run.sh --run training
# atau
./run.sh training

# Jalankan evaluasi model yang baru selesai ditraining
./run.sh --run evaluate
# atau
./run.sh evaluate
```

#### Cara Penggunaan (Windows):
```cmd
# Jalankan antrean training model
run.bat --run training
# atau
run.bat training

# Jalankan evaluasi model yang baru selesai ditraining
run.bat --run evaluate
# atau
run.bat evaluate
```

#### ⚙️ Cara Kerja Otomatisasi:
1. **Mode Training (`training`):**
   - Mendeteksi spesifikasi VRAM GPU utama secara dinamis untuk menyetel `batch size` dan `gradient accumulation steps` yang aman (mencegah CUDA OOM pada GPU 12GB, 16GB, 24GB, atau lebih besar).
   - Membaca antrean eksperimen dari `scripts/training_queue.json`.
   - Jika suatu job mengalami kegagalan, pesan error akan disimpan pada `scripts/parameter_end_training.json`, dan script **tetap melanjutkan** eksekusi job berikutnya dalam antrean.
2. **Mode Evaluasi (`evaluate`):**
   - Membaca daftar training yang sukses dari `scripts/parameter_end_training.json`.
   - Otomatis mencocokkan ketersediaan file hasil inferensi di direktori `eval_results_full/`.
   - Mengecek apakah kombinasi model dan judge terpilih (default: direct DeepSeek API) sudah pernah dievaluasi sebelumnya (untuk menghindari duplikasi biaya API).
   - Jika semua model baru sudah selesai dievaluasi, script akan menampilkan notifikasi:
     `🔔 NOTIFIKASI: Semua evaluasi model sudah dijalankan!`

---

### 3. Pipeline Data Manual (Urutan Eksekusi)
Jalankan script di folder `scripts/` secara berurutan:

1.  **Ekstraksi Teks:** `python 01_pdf_to_markdown_marker.py`
2.  **Pembersihan Teks:** `python 01b_llm_cleaning.py`
3.  **Ingest ke Vector DB:** `python 02_ingest_to_chroma.py`
4.  **Pembuatan Dataset RAFT:** `python 03_generate_dataset.py`
5.  **Training:** `python 04_train_model.py`
6.  **Export:** `python 05_export_to_gguf.py`

### 4. Menjalankan Aplikasi
Pilih antarmuka yang ingin digunakan:

- **Web Dashboard:** `streamlit run app/app.py`
- **API Server:** `uvicorn app.main:app --reload`
- **Interactive CLI:** `python app/app_cli.py`

---

## 📄 Metodologi RAFT & CoT

Sistem ini menggunakan format pesan khusus untuk memastikan akurasi:
- **Konteks Oracle:** Informasi benar dari dokumen.
- **Konteks Distractor:** Informasi pengalih yang mirip namun tidak relevan.
- **Thought Process:** Model dilatih untuk menuliskan langkah logika dalam tag `<thought>` sebelum menjawab.

**Contoh Output:**
```text
<thought>
User menanyakan hukum asuransi syariah. 
Berdasarkan [DOC 1] (Fatwa DSN-MUI No. 21), asuransi syariah menggunakan akad tabarru. 
Saya akan menjelaskan perbedaan akad ini dengan asuransi konvensional.
</thought>

Asuransi syariah dalam Islam diperbolehkan asalkan menggunakan prinsip tolong-menolong (tabarru)...
```

---

## ⚠️ Konfigurasi Tambahan

- **Firestore:** Letakkan file `serviceAccountKey.json` di direktori `app/` untuk mengaktifkan sinkronisasi chat.
- **Ollama:** Pastikan server Ollama berjalan di `localhost:11434` sebelum menjalankan aplikasi atau generator dataset.

---
*Dikembangkan untuk Tugas Akhir/Riset Sistem Pakar Ekonomi Syariah.*
