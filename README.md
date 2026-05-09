# Ekonomi Syariah Chatbot LLM (HANIF)

Proyek ini bertujuan untuk membangun asisten AI spesialis Ekonomi Syariah (HANIF) menggunakan teknik **RAFT (Retrieval-Augmented Fine-Tuning)**. Model dilatih untuk memberikan jawaban yang akurat berdasarkan dokumen referensi dengan proses penalaran (Chain-of-Thought).

## 🚀 Alur Kerja (Pipeline)

### 1. Preprocessing Data
- `scripts/01_pdf_to_markdown_marker.py`: Mengonversi dokumen PDF mentah menjadi format Markdown menggunakan library `marker`.
- `scripts/01b_llm_cleaning.py`: Membersihkan teks hasil konversi menggunakan LLM (Qwen2.5) untuk memperbaiki struktur heading, tabel, dan membuang teks sampah (noise).

### 2. Manajemen Basis Data Vektor
- `scripts/02_ingest_to_chroma.py`: Memecah dokumen menjadi potongan kecil (chunking) dan menyimpannya ke dalam **ChromaDB** menggunakan embedding `multilingual-e5-large`.
- `scripts/02b_check_database_chroma.py`: Alat bantu untuk memverifikasi jumlah data dan isi koleksi di database vektor.

### 3. Pembuatan Dataset (RAFT)
- `scripts/03_generate_dataset.py`: **Inti dari proyek.** Script ini menghasilkan dataset SFT (Supervised Fine-Tuning) dengan format RAFT.
    - Mengambil 1 dokumen asli (Oracle) dan 3 dokumen pengalih (Distractors).
    - Meminta LLM membuat pertanyaan, proses berpikir (`<thought>`), dan jawaban.
- `scripts/03b_create_master_dataset.py`: Menggabungkan berbagai file JSONL hasil generate menjadi satu file `master_dataset.jsonl` untuk pelatihan.

### 4. Pelatihan Model (Fine-Tuning)
- `scripts/04_train_model.py`: Melakukan fine-tuning model (misal: Llama 3 atau Qwen) menggunakan library `Unsloth` untuk efisiensi VRAM. Menggunakan teknik LoRA/QLoRA.
- `scripts/05_export_to_gguf.py`: Mengekspor model hasil latih ke format **GGUF** agar bisa dijalankan di Ollama atau perangkat lokal lainnya.

### 5. Aplikasi & Inferensi
- `app/main.py`: Entry point aplikasi.
- `app/graph.py`: Logika alur percakapan (state machine) menggunakan LangGraph.
- `app/db.py` & `app/app.py`: Integrasi database dan UI (Streamlit).

---

## 🧐 Evaluasi Metodologi (Analisis Script 03)

Berikut adalah penilaian teknis terhadap sistem pembuatan dataset yang digunakan:

### ✅ Kelebihan (Good)
1. **Implementasi RAFT Sesuai Standar:** Penggunaan Oracle dan Distractors melatih model agar tahan terhadap informasi yang tidak relevan (noise) saat RAG.
2. **Chain-of-Thought (CoT):** Adanya tag `<thought>` melatih model untuk melakukan penalaran sebelum menjawab, meningkatkan akurasi logika.
3. **Hard Negatives:** Penggunaan *similarity search* untuk mencari distractor memastikan pengalih bersifat menantang, bukan sekadar teks acak.
4. **Format ShareGPT:** Dataset siap digunakan langsung dengan library populer seperti Unsloth atau Axolotl.

### 💡 Saran Perbaikan (Improve)
1. **Kapasitas Teacher Model:** Penggunaan model 7B untuk generate dataset mungkin kurang tajam logikanya. Disarankan menggunakan model 14B+ atau 32B+ (via API atau GPU besar) untuk hasil dataset yang lebih berkualitas.
2. **Skenario "Tanpa Jawaban":** Tambahkan sekitar 10-20% sampel yang hanya berisi *distractors* tanpa *oracle*, lalu latih model untuk menjawab "Informasi tidak ditemukan". Ini penting untuk mencegah halusinasi.
3. **Variasi Pertanyaan:** Saat ini prompt berfokus pada pertanyaan faktual. Tambahkan instruksi untuk membuat pertanyaan berbasis perbandingan, sintesis, atau analisis kasus ekonomi.
4. **Efisiensi Batching:** Untuk mempercepat proses, satu chunk bisa diminta untuk menghasilkan 2-3 variasi pertanyaan sekaligus dalam satu kali panggil LLM.

---
*Dikembangkan untuk Tugas Akhir - Sistem Pakar Ekonomi Syariah.*