# DRAFT BAB 4 & BAB 5 TUGAS AKHIR — ALIF FATURROHMAN

# BAB 4 — HASIL DAN PEMBAHASAN

Bab ini menyajikan hasil dari setiap tahapan penelitian yang telah dilaksanakan, mulai dari penyiapan data hingga evaluasi performa model menggunakan framework RAGAS. Pembahasan dilakukan untuk menganalisis efektivitas setiap strategi *Retrieval-Augmented Fine-Tuning* (RAFT) dalam meningkatkan literasi ekonomi syariah.

## 4.1 Hasil Penelitian

### 4.1.1 Hasil Pre-processing Data
Tahap pre-processing dilakukan terhadap **10 dokumen PDF** utama yang bersumber dari publikasi resmi OJK dan KNEKS. Dokumen-dokumen ini mencakup berbagai topik krusial seperti ekonomi halal, prinsip dasar ekonomi syariah, hingga manajemen kekayaan Islam. Proses konversi menghasilkan **12 file Markdown** yang telah dibersihkan dari elemen administratif seperti nomor halaman dan header yang repetitif.

Hasil pre-processing ini selanjutnya diingest ke dalam basis data vektor menggunakan model embedding **`intfloat/multilingual-e5-large`**. Pemilihan model ini didasarkan pada kemampuannya dalam menangani teks multilingual (termasuk Bahasa Indonesia) dengan akurasi semantik yang tinggi, yang sangat penting untuk pencarian rujukan ekonomi syariah yang presisi.

### 4.1.2 Hasil Penyusunan Dataset RAFT
Penelitian ini menghasilkan dataset RAFT (*Retrieval-Augmented Fine-Tuning*) sebanyak **1.454 entri** dalam format Alpaca JSON. Dataset ini dirancang untuk melatih model agar mampu melakukan penalaran mendalam (*reasoning*) berdasarkan dokumen referensi. Setiap entri dataset memuat:
1.  **Oracle Document:** Potongan teks (chunk) yang berisi jawaban langsung atas pertanyaan.
2.  **Distractor Document:** Dokumen tambahan yang tidak relevan untuk menguji kemampuan filter model.
3.  **Chain-of-Thought (CoT):** Analisis langkah-demi-langkah yang dibungkus dalam tag `<thought>`.

Topik yang dicakup dalam dataset ini sangat komprehensif, meliputi:
- Dasar Ekonomi dan Industri Halal.
- Transaksi yang Dilarang dalam Islam (Riba, Gharar, Maysir).
- Keuangan Sosial Islam (Zakat, Infak, Sedekah, dan Wakaf).
- Industri Keuangan Syariah (Perbankan dan Non-Perbankan).
- Etika Bisnis dan Manajemen Kekayaan Syariah.

Dataset ini kemudian dibagi menjadi tiga strategi (Strat 1, 2, dan 3) dan dievaluasi menggunakan subset uji (*test set*) sebanyak 291 query unik untuk memastikan keandalan model pada data yang belum pernah dilihat sebelumnya.

### 4.1.3 Hasil Fine-Tuning Model (Implementasi QLoRA)
Fine-tuning dilakukan melalui tiga iterasi utama untuk menemukan konfigurasi optimal. Hasil evolusi parameter training dirangkum pada Tabel 4.1.

**Tabel 4.1 — Evolusi Konfigurasi Hyperparameter Training**

| Sesi Training | Status | Parameter Kunci | Hasil Temuan |
| :--- | :--- | :--- | :--- |
| **V1 (Awal)** | Gagal | LR 2e-4, Rank 16, Ep 1 | Terjadi *Model Collapse* pada Strat 1 (output *gibberish*). |
| **V2 (Retrain)** | Sukses | LR 5e-5, Rank 16, Ep 2 | Model sembuh, penalaran CoT mulai muncul secara koheren. |
| **V3 (Final)** | Optimal | **LR 2e-5, Rank 32, Ep 3** | Akurasi maksimal dengan kapasitas memori LoRA yang lebih tinggi. |

Selama proses training V3 (Final), dilakukan optimasi manajemen memori untuk mengatasi kendala *CUDA Out of Memory* (OOM). Meskipun target *Batch Size* di proposal adalah 2, sistem diubah menggunakan **Batch Size 1 dengan Gradient Accumulation 8** (Total Batch 8 tetap terjaga) untuk menyesuaikan dengan kapasitas VRAM 12GB pada GPU NVIDIA RTX 3060.

### 4.1.3 Hasil Evaluasi Kualitas Jawaban (Metrik RAGAS)
Kualitas jawaban model diukur menggunakan juri eksternal **DeepSeek v4 Flash via OpenRouter** untuk menjaga objektivitas penilaian. Skor rata-rata dari 291 query per strategi disajikan pada Tabel 4.2.

**Tabel 4.2 — Perbandingan Performa Strategi (Full Dataset Evaluation)**

| Metrik Ragas | Strat 1 (Pure Aug) | Strat 2 (Cross) | Strat 3 (Dual Sync) | Pemenang |
| :--- | :---: | :---: | :---: | :--- |
| **Faithfulness** | 0.438 | **0.632** | 0.399 | **Strat 2** |
| **Answer Relevancy** | 0.920 | **0.948** | 0.907 | **Strat 2** |
| **Context Precision** | 0.996 | 0.993 | **1.000** | **Strat 3** |
| **Context Recall** | 0.448 | **0.809** | 0.439 | **Strat 2** |

---

## 4.2 Pembahasan

### 4.2.1 Superioritas Strategi Cross (Strat 2)
Berdasarkan hasil evaluasi pada Tabel 4.2, **Strategi 2 (Cross)** terbukti sebagai metode paling efektif dalam domain ekonomi syariah. Keunggulan mencolok terlihat pada metrik *Context Recall* yang mencapai **0.809** (81%). Hal ini menunjukkan bahwa melatih model dengan pola penalaran lintas dokumen memungkinkan LLM untuk menangkap informasi kunci dari teks rujukan secara jauh lebih lengkap dibandingkan strategi lainnya. 

### 4.2.2 Analisis Fenomena Model Collapse dan Peran Learning Rate
Eksperimen pada Sesi V1 menunjukkan bahwa penggunaan *Learning Rate* (LR) yang terlalu tinggi (`2e-4`) pada dataset yang sangat spesifik dapat menyebabkan kegagalan total pada bobot model (*Catastrophic Forgetting*). Fenomena ini ditandai dengan munculnya karakter acak (*gibberish*) seperti `scsc` dan `BigInteger`. 

Pembahasan ini menekankan bahwa stabilitas model pada tugas khusus seperti Ekonomi Syariah lebih bergantung pada laju belajar yang lambat dan stabil (`2e-5`) serta jumlah iterasi (Epoch) yang cukup, daripada sekadar volume data. Hal ini dibuktikan pada Strat 3 yang meskipun memiliki volume data terbesar, justru tertinggal dalam hal *Faithfulness* karena kompleksitas campuran data *raw* dan *augmented* yang membutuhkan regulasi parameter lebih ketat.

### 4.2.3 Objektivitas Penilaian: DeepSeek vs Juri Lokal
Penelitian ini menemukan adanya perbedaan signifikan antara juri lokal (Qwen 7B) dan juri API (DeepSeek v4 Flash). DeepSeek ditemukan jauh lebih ketat (*strict*) dalam menilai kepatuhan terhadap fakta (*Faithfulness*), dengan penurunan skor rata-rata sebesar 15-20% dibanding juri lokal. Penggunaan juri eksternal yang lebih cerdas ini memberikan validasi yang lebih kuat terhadap klaim akurasi model dalam Tugas Akhir.


# BAB 5 — KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

## 5.2 Saran
