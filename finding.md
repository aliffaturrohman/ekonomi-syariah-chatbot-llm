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

## 9. Model Fine-Tune (GGUF) & Protokol Stress Test
*Update: Senin, 22 Juni 2026*

### Model Fine-Tune & Perbandingan Parameter (Strat 2 Cross-document)

Semua model adalah LoRA adapter base **Qwen2.5-7B-Instruct**, target modules: k_proj, o_proj, v_proj, up_proj, down_proj, q_proj, gate_proj (7 mods).

| Model | r | lora_alpha | LR | Ep | Train Loss | Runtime | Status |
|:---|---|---:|---:|---:|---:|:---:|:---:|
| Strat 2 LR 5e-5 Ep 2 (yg skrg di config) | 16 | 16 | 5e-5 | 2 | **1.686** | 3757s | ✅ Ada GGUF |
| Strat 2 Final R32 Ep 3 | 32 | 64 | 2e-5 | 3 | **1.582** (↓6%) | 5690s | ✅ Ada adapter |
| **Strat 2 Final R32 Ep 4** ✅ **Terbaik** | **32** | **64** | **2e-5** | **4** | **1.450 (↓14%)** | **7492s** | ✅ Ada adapter |

**Rekomendasi:** Strat 2 Final R32 Ep 4 — parameter paling tinggi (r=32, alpha=64, ep=4) dengan loss terendah 1.450. Base model Qwen2.5-7B-Instruct.

⚠️ **Catatan:** Model ini belum di-GGUF. Untuk stress test sekarang (tanpa Ollama), API pake OpenRouter via `LLM_PROVIDER=openrouter`. Kalau mau GGUF jalanin `05_export_to_gguf.py` dulu. Detail adapter ada di:
`models/adapters/qwen_raft_ekonomi_syariah_strat2_final_r32_lr2e5_ep4_20260609_1552/`

### Cara Impor ke Ollama

```bash
cat > /tmp/Modelfile << 'EOF'
FROM /home/alif-faturrohman/coding/ta_ekonomi_syariah/ekonomi-syariah-chatbot-llm/models/adapters/qwen_raft_ekonomi_syariah_strat2_lr5e5_ep2_20260602_1802_gguf/Qwen2.5-7B-Instruct.Q4_K_M.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.0
PARAMETER num_ctx 8192
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
SYSTEM """Anda adalah HANIF, asisten AI Ekonomi Syariah. Jawablah pertanyaan pengguna dengan akurat berdasarkan konteks yang diberikan."""
EOF

ollama create qwen-ekonomi-syariah-strat2 -f /tmp/Modelfile
ollama run qwen-ekonomi-syariah-strat2 "Apa itu ekonomi syariah?"
```

### vLLM Server (Sedang Berjalan)

vLLM sudah aktif di **port 8001** — Strat 2 Final R32 Ep 4 + LoRA adapter.

| Item | Detail |
|:---|---|
| Endpoint | `http://localhost:8001/v1` |
| Model | `qwen-strat2` |
| Base model | Qwen2.5-7B-Instruct AWQ 4bit |
| LoRA | `models/strat2-final-lora/` (symlink) |
| GPU | RTX 3060 12GB (mem utilization 80%) |

```bash
# Test langsung
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-strat2","messages":[{"role":"user","content":"Apa itu ekonomi syariah?"}]}'
```

### Stress Test — Cara Pakai

File: `chatbot-api-ekonomi-syariah/stress_test.py`

```bash
export LLM_PROVIDER=vllm
export BEARER_TOKEN="eyJhbG..."
export API_KEY="mantapAnjing"

cd chatbot-api-ekonomi-syariah
python stress_test.py
```

Protokol: incremental load 1-10 users | 2 skenario (identical/variative) | 5 iterasi
Metrik: latency (ms), throughput (TPS), error rate (%)
Dataset: 1454 pertanyaan dari MASTER_RAFT_DATASET.jsonl
Endpoint: vLLM direct (skip auth) — http://localhost:8001
Konfigurasi: max_model_len=8192, max_tokens=2048, GPU mem 85%

### Hasil Stress Test (vLLM + Strat 2 Final R32 Ep 4)

| Skenario | Concurrent | Lat p50 | Lat p95 | Lat p99 | TPS | Error Rate | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| identical | 1 | 8.02s | 8.02s | 8.02s | 63.3 | 0.0% | 2,540 |
| identical | 2 | 8.31s | 8.31s | 8.31s | 61.1 | 0.0% | 5,080 |
| identical | 3 | 8.36s | 8.36s | 8.36s | 60.1 | 0.0% | 7,620 |
| identical | 4 | 8.43s | 8.43s | 8.43s | 59.3 | 0.0% | 10,160 |
| identical | 5 | 8.53s | 8.53s | 8.53s | 58.5 | 0.0% | 12,700 |
| identical | 6 | 8.60s | 8.60s | 8.60s | 57.8 | 0.0% | 15,240 |
| identical | 7 | 8.67s | 8.67s | 8.67s | 58.0 | 0.0% | 17,605 |
| identical | 8 | 8.90s | 8.91s | 8.91s | 57.1 | 0.0% | 20,320 |
| identical | 9 | 9.95s | 9.95s | 9.96s | 51.1 | 0.0% | 22,860 |
| identical | 10 | 10.04s | 10.04s | 10.04s | 50.6 | 0.0% | 25,400 |
| variative | 1 | 16.05s | 16.09s | 16.09s | 62.8 | 0.0% | 4,776 |
| variative | 2 | 12.97s | 16.39s | 16.39s | 61.3 | 0.0% | 4,776 |
| variative | 3 | 13.11s | 16.97s | 16.97s | 60.2 | 0.0% | 7,164 |
| variative | 4 | 12.87s | 16.75s | 17.01s | 59.3 | 0.0% | 9,552 |
| variative | 5 | 13.17s | 17.30s | 17.31s | 58.5 | 0.0% | 11,940 |
| variative | 6 | 13.00s | 16.86s | 16.87s | 57.8 | 0.0% | 14,328 |
| variative | 7 | 12.12s | 16.97s | 17.01s | 57.6 | 0.0% | 24,481 |
| variative | 8 | 12.00s | 17.11s | 17.15s | 56.8 | 0.0% | 27,060 |
| variative | 9 | 13.15s | 19.09s | 19.10s | 53.2 | 0.0% | 32,490 |
| variative | 10 | 12.99s | 19.46s | 19.48s | 51.9 | 0.0% | 35,905 |

### Hasil Output Files

| File | Description |
|:---|---|
| `chatbot-api-ekonomi-syariah/stress_test_results_full.json` | Data raw JSON (51K) |
| `chatbot-api-ekonomi-syariah/stress_test_results_full.csv` | Tabel ringkasan |
| `chatbot-api-ekonomi-syariah/stress_test_results_full_iterations.csv` | Data per-iterasi (raw) |
| `chatbot-api-ekonomi-syariah/stress_test_results_full.png` | Chart latency + TPS + error rate |

### Analisis Hasil

1. **Error rate 0%** di semua skenario — tidak ada timeout, OOM, atau HTTP error pada RTX 3060 12GB
2. **Identical request** (cache-friendly): latency p50 stabil 8-10s sepanjang 1-10 user. Naik 25% dari 8s ke 10s.
3. **Variative request** (real-world): latency p50 ~12-16s. p95 meningkat dari 16s ke 19s seiring concurrency.
4. **Throughput (TPS)**: 63 → 52 (penurunan 17% dari 1 ke 10 user). GPU hampir full di 10 user.
5. **Throat**: Concurrency 9-10 menunjukkan TPS turun agak tajam (51-53) — batas optimal RTX 3060.
6. **Total tokens generated**: ~408K tokens across all tests, 0 errors.

## 7. Temuan Eksperimen Rank 64 (Sesi V4)
*Update: Selasa, 9 Juni 2026, 13:45 WIB*
- **Rank 64 vs Rank 32:** 
    - **Strat 1 (Pure Aug):** Berhasil diselesaikan dengan Rank 64. Peningkatan Rank memberikan kapasitas memori yang lebih besar untuk data hasil augmentasi.
    - **Strat 2 & 3 (Cross & Dual):** Mengalami **CUDA Out of Memory (OOM)**. 
- **Analisis Memori:** Strategi yang melibatkan konteks dokumen panjang (Cross-document & Dual) tidak kompatibel dengan LoRA Rank 64 pada GPU VRAM 12GB (RTX 3060). Beban memori untuk menyimpan gradien pada rank yang lebih tinggi melampaui sisa VRAM yang tersedia setelah alokasi model dasar dan konteks.
- **Kesimpulan:** Rank 32 adalah batas optimal untuk stabilitas training di environment hardware saat ini untuk seluruh strategi.

## 8. Temuan Eksperimen Epoch 4 (Sesi V5)
*Update: Selasa, 9 Juni 2026, 19:45 WIB*
- **Strat 1 & 2 (Rank 32, Ep 4):** **SUKSES**. Penambahan epoch menjadi 4 meningkatkan pemahaman model pada dataset Pure Augmentation dan Cross-document.
- **Strat 3 (Rank 32, Ep 4):** Mengalami **CUDA Out of Memory (OOM)**. 
- **Analisis:** Dataset Strat 3 (Dual Sync) memiliki jumlah baris dua kali lipat (~2300 baris) dan rata-rata panjang konteks yang lebih besar. Meskipun Rank sudah diturunkan ke 32, kombinasi Epoch 4 dan Context 2048 melebihi batas 12GB VRAM saat proses evaluasi internal/checkpointing.
