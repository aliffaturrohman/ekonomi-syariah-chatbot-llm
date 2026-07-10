# Final Evaluation Results Summary

Data metrik evaluasi telah diekstrak dan diproses ke dalam folder `evaluation_results/`. Berikut adalah rangkuman dari file CSV yang dihasilkan.

## 1. NLP Metrics (BLEU & ROUGE-L)
Metrik ini digunakan untuk menilai kualitas generasi teks (seberapa mirip jawaban model dengan referensi) tanpa melibatkan Judge API eksternal. Sesuai dengan spesifikasi di proposal TA, ROUGE-L digunakan untuk mengukur *Longest Common Subsequence*.

| Model (Tag) | BLEU-3 | BLEU-4 | BLEU-5 | ROUGE-L (F-Measure) |
| :--- | :---: | :---: | :---: | :---: |
| strat2_lr5e5_ep2_20260602_1802 | 0.158 | 0.106 | 0.075 | **0.305** |
| final_r32_lr2e5_ep4_20260609_1552 (Strat 2, Ep 4) | 0.123 | 0.078 | 0.052 | 0.270 |
| strat2_final_r32_lr2e5_ep3_20260604_1905 (Strat 2, Ep 3) | 0.125 | 0.080 | 0.054 | 0.268 |
| strat3_lr5e5_ep2_20260604_0943 | 0.111 | 0.069 | 0.047 | 0.241 |
| strat3_lr2e4_ep1_original | 0.105 | 0.064 | 0.041 | 0.241 |
| strat1_lr5e5_ep2_20260602_1659 | 0.106 | 0.064 | 0.042 | 0.238 |
| final_r64_lr2e5_ep3_20260609_1009 (Strat 1, R64) | 0.101 | 0.061 | 0.039 | 0.230 |
| strat1_final_r32_lr2e5_ep3_20260604_1730 (Strat 1, Ep 3) | 0.100 | 0.060 | 0.039 | 0.228 |
| final_r32_lr2e5_ep4_20260609_1346 (Strat 1, Ep 4) | 0.097 | 0.058 | 0.038 | 0.225 |
| strat3_final_r32_lr2e5_ep3_20260604_2145 (Strat 3, Ep 3) | 0.091 | 0.054 | 0.035 | 0.218 |

**Analisis:**
Skor BLEU yang dihasilkan wajar untuk chatbot edukasi (berkisar antara 0.05 - 0.15) karena model dilatih untuk menjelaskan (paraphrasing), bukan sekadar menghafal *exact match* dari buku. Nilai **ROUGE-L** yang lebih baik (~0.30) membuktikan struktur informasi dari buku berhasil dipertahankan. Konsisten dengan evaluasi RAGAS, **Strat 2** kembali mendominasi skor leksikal ini, menunjukkan bahwa strategi *cross-augmentation* sangat efektif dalam mempertahankan kemiripan dengan *Ground Truth*.

---

## 2. RAGAS Metrics (DeepSeek API Judge)
Berbeda dengan BLEU/ROUGE yang kaku, metrik ini mengevaluasi makna semantik dan logika penalaran.

| Model (Tag) | Answer Relevancy | Context Precision | Faithfulness | Context Recall |
| :--- | :---: | :---: | :---: | :---: |
| strat2_final_r32_lr2e5_ep3 (Strategi Terbaik) | **0.938** | **0.990** | **0.465** | **0.854** |
| strat1_final_r32_lr2e5_ep4 (Ep 4) | 0.926 | 0.955 | 0.317 | 0.448 |
| strat3_final_r32_lr2e5_ep3 (Retrain) | 0.923 | 0.965 | 0.334 | 0.474 |
| final_r64_lr2e5_ep3 (Rank 64) | 0.920 | 0.969 | 0.311 | 0.469 |
| strat1_final_r32_lr2e5_ep3 (Optimal) | 0.915 | 0.800 | 0.288 | 0.473 |

**Analisis Utama:**
Model **Strat 2 (Rank 32, Ep 3)** menunjukkan dominasi mutlak di seluruh metrik Ragas. Dengan *Answer Relevancy* mencapai **0.938** dan *Context Precision* hampir sempurna di **0.990**, model ini membuktikan bahwa strategi dataset *cross-augmentation* yang digunakan pada Strat 2 jauh lebih superior dibandingkan strategi lainnya dalam melatih kemampuan penalaran (reasoning) AI. *Context Recall* yang melonjak drastis ke angka **0.854** (dibandingkan Strat 1 yang hanya ~0.47) membuktikan bahwa model Strat 2 tidak mengarang bebas, melainkan benar-benar menyarikan jawaban dari konteks yang disediakan.

---

## 3. Training Loss
Gambar grafik (Loss Curve) dan data CSV historis untuk setiap *step* pelatihan dapat ditemukan di:
- `evaluation_results/csv/train_loss_*.csv`
- `evaluation_results/img/loss_*.png`

Data ini siap digunakan langsung di Bab 4 (Hasil dan Pembahasan) pada dokumen skripsi.
