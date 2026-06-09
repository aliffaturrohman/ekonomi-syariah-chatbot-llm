# Temuan Evaluasi & Model (finding.md)
*Update terakhir: Kamis, 4 Juni 2026, 17:40 WIB*

## 1. Perbandingan Performa Strategi (Full Dataset - DeepSeek v4 Flash)
Evaluasi final menggunakan juri eksternal **DeepSeek v4 Flash via OpenRouter** terhadap model perbaikan (LR 5e-5, Ep 2).

| Metrik | Strat 1 (Pure Aug) | Strat 2 (Cross) | Strat 3 (Dual Sync) | Pemenang |
| :--- | :---: | :---: | :---: | :--- |
| **Faithfulness** | 0.438 | **0.632** | 0.399 | **Strat 2** |
| **Answer Relevancy** | 0.920 | **0.948** | 0.907 | **Strat 2** |
| **Context Precision** | 0.996 | 0.993 | **1.000** | **Strat 3** |
| **Context Recall** | 0.448 | **0.809** | 0.439 | **Strat 2** |

### **Analisis Temuan Penting:**
- **Kredibilitas Juri:** DeepSeek v4 Flash terbukti jauh lebih **strict (ketat)** dibandingkan Qwen lokal. Skor Faithfulness turun rata-rata 15-20%, memberikan data yang lebih objektif untuk TA.
- **Superioritas Strategi 2:** Metode *Cross-document reasoning* tetap menjadi yang terbaik dalam menjaga relevansi dan ingatan konteks (Recall 80%).
- **Isu Token Ragas:** Evaluasi Ragas sangat boros token karena proses ekstraksi pernyataan yang berulang. Biaya per strategi mencapai ~$0.8 - $1.0.

## 2. Kendala Teknis & Solusi (Environment Python 3.14)
- **Isu Kematian Proses (Serialization Error):** Ditemukan kegagalan pada library `datasets` saat proses *pickling* di Python 3.14. 
    - *Gejala:* `TypeError: _Pickler._batch_setitems() missing 1 required positional argument`.
    - *Solusi:* Dilakukan **Hot-fix / Patching** langsung pada core library di `site-packages/datasets/utils/_dill.py` untuk menyesuaikan signature fungsi.
- **Isu Memori (CUDA Out of Memory):** Training dengan **Rank 32** dan **Context 2048** melebihi batas 12GB jika Batch Size = 2.
    - *Solusi:* Konfigurasi diubah menjadi **Batch Size 1** dengan **Gradient Accumulation 8** (Total Batch 8 tetap terjaga).

## 3. Evolusi Konfigurasi Training
| Sesi | Status | Parameter Kunci | Hasil |
| :--- | :--- | :--- | :--- |
| **V1 (Awal)** | Gagal | LR 2e-4, Rank 16, Ep 1 | Model Collapse (Strat 1) / OOT |
| **V2 (Retrain)** | Sukses | LR 5e-5, Rank 16, Ep 2 | Model sembuh, penalaran muncul |
| **V3 (Final)** | *Running* | LR 2e-5, **Rank 32**, **Ep 3** | Target akurasi maksimal TA |

## 5. Kendala Training Final (V3)
*Update: Kamis, 4 Juni 2026, 21:33 WIB*
- **Status Strat 1 & 2:** Berhasil diselesaikan dengan Rank 32, Ep 3.
- **Status Strat 3:** Mengalami **CUDA Out of Memory (OOM)** di awal proses training.
- **Analisis OOM:** Dataset Strat 3 (Dual) memiliki baris dengan konteks yang sangat panjang, sehingga saat proses alokasi memori untuk *backpropagation* Rank 32, penggunaan VRAM melonjak melampaui batas 11.6 GB.
- **Tindakan Lanjutan:** 
    1. Melakukan **Retry** untuk Strat 3 dengan parameter yang sama persis (Rank 32, Max Seq 2048) untuk memastikan apakah kegagalan bersifat fluktuatif atau permanen.
    2. Jika tetap gagal OOM, akan dilakukan penurunan `MAX_SEQ_LENGTH` menjadi 1024 sebagai solusi cadangan.


## 6. Pembaruan Virtual Environment & Solusi Disk Quota Exceeded
*Update: Selasa, 9 Juni 2026, 06:34 WIB*
- **Rebuild Environment:** Virtual environment (`venv`) telah dihapus dan dibangun ulang dari nol secara bersih menggunakan Python 3.14.4 untuk memastikan konsistensi seluruh paket dependensi.
- **Kendala Disk Quota Exceeded (OSError 122):** 
    - *Gejala:* Instalasi terhenti di tengah jalan dengan error `OSError: [Errno 122] Disk quota exceeded`, meskipun partisi SSD utama masih memiliki sisa ruang fisik sebesar 579 GB.
    - *Analisis:* Sistem membatasi ukuran direktori `/tmp` bawaan Linux (yang dipasang sebagai RAM/tmpfs) hanya sebesar **7,7 GB** dengan batasan user quota (`usrquota`). Ekstraksi dan instalasi paket-paket Machine Learning raksasa (seperti PyTorch dan CUDA) melebihi batas tersebut.
    - *Solusi:* Proses instalasi dialihkan menggunakan folder temporary buatan sendiri di dalam direktori proyek (`tmp_pip`) dengan menyetel variabel lingkungan `TMPDIR` sebelum menjalankan perintah instalasi. Folder `tmp_pip` berhasil dihapus setelah instalasi selesai untuk menghemat ruang.
- **Patch Kompatibilitas Python 3.14 (Auto-Patcher):**
    - Agar kompatibilitas ini tidak hilang/terhapus ketika proyek dipindahkan atau di-clone ke komputer lain (misalnya server GPU yang lebih besar), kami telah membuat skrip otomatisasi **[patch_datasets.py](file:///home/alif-faturrohman/coding/ekonomi-syariah-chatbot-llm/scripts/patch_datasets.py)**.
    - Skrip ini akan dipanggil secara otomatis di awal eksekusi program training (`04_train_model.py`) dan evaluasi (`07_run_ragas_evaluation.py`). Skrip akan memeriksa apakah file `_dill.py` di dalam virtual environment lokal memiliki signature yang tepat, lalu menerapkannya secara dinamis menggunakan parameter variadik (`*args, **kwargs`). Hashing dataset kini berjalan secara portabel tanpa memerlukan intervensi manual lagi.
