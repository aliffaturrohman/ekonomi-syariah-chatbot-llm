# Hasil Training Model Ekonomi Syariah

Berikut adalah ringkasan parameter model yang telah di-training, status keberhasilan, dan keterangan tambahan terkait hasil training.

## Tabel Ringkasan Training

| No | Strategi | Learning Rate | Epoch | Rank (LoRA) | Status | Ragas Eval | Selesai Pada | Keterangan |
|:---:|:---|:---:|:---:|:---:|:---|:---:|:---|:---|
| 1 | Strat 1 | 2e-4 | 1 | 16 | ❌ Gagal | - | Sesi V1 | Catastrophic Collapse (Gibberish) |
| 2 | Strat 3 | 2e-4 | 1 | 16 | ✅ Selesai | ✅ 🦙 | Sesi V1 | Original V1 |
| 3 | Strat 1 | 5e-5 | 2 | 16 | ✅ Selesai | ✅ 🦙 🌐 | Sesi V2 | Sembuh, Penalaran CoT Muncul |
| 4 | Strat 2 | 5e-5 | 2 | 16 | ✅ Selesai | ✅ 🦙 🌐 | Sesi V2 | Sembuh, Penalaran CoT Muncul |
| 5 | Strat 3 | 5e-5 | 2 | 16 | ✅ Selesai | ✅ 🌐 | Sesi V2 | Sembuh, Penalaran CoT Muncul |
| 6 | Strat 1 | 2e-5 | 3 | 32 | ✅ Sukses | ✅ 🐉 | Sesi V3 | - |
| 7 | Strat 2 | 2e-5 | 3 | 32 | ✅ Sukses | - | Sesi V3 | **Strategi Terbaik** |
| 8 | Strat 3 | 2e-5 | 3 | 32 | ✅ Sukses | - | Sesi V3 | Berhasil setelah Retry |
| 9 | Strat 1 | 2e-5 | 3 | 64 | ✅ Sukses | - | 2026-06-09 | Sesi V4: Eksperimen Rank Tinggi |
| 10 | Strat 1 | 2e-5 | 4 | 32 | ✅ Sukses | - | 2026-06-09 | Sesi V5: Eksperimen Epoch Tinggi |
| 11 | Strat 2 | 2e-5 | 4 | 32 | ✅ Sukses | - | 2026-06-09 | Sesi V5: Eksperimen Epoch Tinggi |

## Legenda Ragas Eval
- 🦙: Evaluasi menggunakan Judge **Ollama (Qwen)**
- 🌐: Evaluasi menggunakan Judge **OpenRouter (DeepSeek V4 Flash)**
- 🐉: Evaluasi menggunakan Judge **DeepSeek API (Direct)**
- ✅: Evaluasi selesai dan metrik tersedia di `eval_metrics/`

## Catatan Tambahan

### Error & Troubleshooting
- **CUDA Out of Memory (OOM):** Terjadi pada sesi retrain Strat 3 (final). Log: `retrain_final.log`.
- **Solusi OOM:** Dilakukan retry dengan pembersihan VRAM yang lebih agresif. Berhasil diselesaikan pada `retrain_final_retry.log`.
- **Collapse pada Strat 1 (V1):** LR 2e-4 terlalu tinggi untuk Strat 1, menyebabkan model kehilangan kemampuan bahasa (gibberish). LR diturunkan ke 5e-5 dan 2e-5 pada sesi berikutnya untuk memperbaiki ini.

### Hasil Evaluasi Terkait
Data evaluasi lengkap untuk model-model di atas dapat ditemukan pada folder:
- `ekonomi-syariah-chatbot-llm/eval_metrics/` (Metrik CSV)
- `ekonomi-syariah-chatbot-llm/eval_results_full/` (Detail Jawaban JSON)

---
*Dibuat otomatis oleh Gemini CLI berdasarkan analisis log dan config.*
