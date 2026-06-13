**BAB 4 — HASIL DAN IMPLEMENTASI**  
Bab ini memaparkan hasil dari setiap tahapan implementasi sistem yang telah dirancang pada Bab 3, mulai dari pre-processing data hingga deployment sistem chatbot edukatif berbasis RAFT. Seluruh proses dilaksanakan sesuai dengan kerangka kerja ADDIE pada tahap *Development* dan  *Implementation*.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAMUlEQVR4nO3WAQkAIBAEsBPMYs4PZhMDWMAA5njYUmxU1UqyAwBAF2cmeZE4AIBO7gentgXapSWpbgAAAABJRU5ErkJggg==)  
**4.1 Lingkungan Implementasi**  
Seluruh proses dalam penelitian ini — mulai dari *pre-processing* data,  *fine-tuning* model, hingga  *deployment* layanan API — dilaksanakan pada satu perangkat yang sama, yaitu komputer lokal milik peneliti. Hal ini sesuai dengan batasan masalah pada Subbab 1.3 yang menetapkan efisiensi komputasi pada perangkat dengan sumber daya terbatas sebagai salah satu kriteria sistem.  
**4.1.1 Spesifikasi Perangkat Keras**  
**Tabel 4.1 — Spesifikasi Perangkat Keras**  
| | |  
|-|-|  
| **Komponen** | **Spesifikasi** |   
| **Perangkat** | [Nama perangkat, misal: Laptop ASUS ROG Strix G15 / PC Rakitan] |   
| **CPU** | [Nama CPU, misal: AMD Ryzen 7 5800H / Intel Core i7-12700H] |   
| **Jumlah Core / Thread** | [X Core / X Thread] |   
| **RAM** | [X GB, misal: 16 GB DDR4 3200MHz] |   
| **GPU** | [Nama GPU, misal: NVIDIA GeForce RTX 3060 Laptop] |   
| **VRAM** | [X GB, misal: 6 GB GDDR6] |   
| **Penyimpanan** | [Tipe dan kapasitas, misal: 512 GB NVMe SSD] |   
| **Sistem Operasi** | [misal: Windows 11 Home 64-bit / Ubuntu 22.04 LTS] |   
   
***Catatan:*** * GPU yang digunakan termasuk dalam kategori GPU konsumen seri NVIDIA RTX, konsisten dengan batasan desain sistem pada Subbab 3.2.3 yang mensyaratkan efisiensi pada GPU konsumen (NVIDIA RTX seri 30, 40, atau 50).*  
**4.1.2 Spesifikasi Perangkat Lunak dan Library**  
**Tabel 4.2 — Spesifikasi Perangkat Lunak Utama**  
| | | |  
|-|-|-|  
| **Kategori** | **Nama** | **Versi** |   
| **Runtime** | Python | [X.X.X, misal: 3.11.9] |   
| **CUDA Toolkit** | CUDA | [X.X, misal: 12.1] |   
| **Framework Deep Learning** | PyTorch | [X.X.X] |   
| **Framework Fine-Tuning** | Unsloth | [X.X.X] |   
| **Library LLM** | Transformers (HuggingFace) | [X.X.X] |   
| **Framework PEFT** | PEFT | [X.X.X] |   
| **Inference Engine** | Ollama | [X.X.X] |   
| **Framework API** | FastAPI | [X.X.X] |   
| **Orkestrasi LLM** | LangChain | [X.X.X] |   
| **Vector Database** | ChromaDB | [X.X.X] |   
| **Database Realtime** | Firebase Admin SDK | [X.X.X] |   
| **Framework Evaluasi RAG** | RAGAS | [X.X.X] |   
| **Load Testing** | Locust | [X.X.X] |   
| **LLM Judge** | DeepSeek (via API) | [nama model, misal: deepseek-chat] |   
   
***Catatan:*** * Seluruh versi library dapat diverifikasi melalui file * *requirements.txt* * yang disertakan pada repositori kode sumber penelitian ini (lihat Lampiran).*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OMQ0AIAwAwdIgBKl1gjacsGCAiZDcTT9+q6oRETMAAPjF6ify6QYAADdyA9Y0AypN+bdfAAAAAElFTkSuQmCC)  
**4.2 Hasil Pre-processing Data**  
Tahap pre-processing dilakukan terhadap 10 dokumen PDF yang bersumber dari publikasi resmi OJK dan KNEKS, mencakup buku teks dan modul pembelajaran terkait ekonomi syariah. Proses ini menghasilkan total XXX chunk teks yang bersih dan terstruktur, dengan rata-rata panjang XXX token per chunk.  
Distribusi dokumen berdasarkan sumber disajikan pada Tabel 4.3 berikut.  
**Tabel 4.3 — Statistik Hasil Pre-processing Data**  
| | | | |  
|-|-|-|-|  
| **Sumber Dokumen** | **Jumlah Dokumen** | **Jumlah Chunk** | **Rata-rata Token/Chunk** |   
| Buku Teks OJK |   |   |   |   
| Modul KNEKS |   |   |   |   
| Artikel Ilmiah |   |   |   |   
| Total |   |   |   |   
   
Seluruh chunk telah melewati proses *noise cleaning* untuk menghilangkan elemen tidak relevan seperti  *header*,  *footer*, nomor halaman, dan karakter simbol. Hasil akhir disimpan dalam format JSON Array sebagaimana dijelaskan pada Subbab 3.2.5, dan selanjutnya diingest ke dalam  *Chroma Vector Database* menggunakan model embedding **[nama embedding model yang digunakan, contoh: all-MiniLM-L6-v2]**.  
Screenshot chunking  
**4.3 Hasil Penyusunan Dataset RAFT**  
Dataset RAFT yang dihasilkan terdiri dari **[X] entri** dalam format Alpaca JSON, yang digunakan sebagai data latih pada tahap fine-tuning. Setiap entri memuat satu *oracle document*, **[X] distractor document**, serta respons berbasis *Chain-of-Thought* (CoT).  
Proses pembuatan pertanyaan (*question generation*) dilakukan secara otomatis menggunakan model Qwen-7B, menghasilkan **[X] pertanyaan unik** yang mencakup topik-topik utama dalam cakupan materi (lihat Subbab 1.3). Sebanyak  **[X]%** entri dihasilkan sepenuhnya otomatis, sementara  **[X]%** sisanya dilakukan revisi manual untuk memastikan kualitas CoT dan relevansi jawaban.  
**Tabel 4.4 — Statistik Dataset RAFT**  
| | |  
|-|-|  
| **Atribut** | **Nilai** |   
| Total entri dataset | [X] |   
| Jumlah *oracle document* per entri | 1 |   
| Jumlah *distractor document* per entri | [X] |   
| Rata-rata panjang jawaban (token) | [X] |   
| Persentase entri di-generate otomatis | [X]% |   
| Persentase entri dengan revisi manual | [X]% |   
   
Contoh representasi satu entri dataset RAFT dalam format Alpaca JSON yang sesungguhnya (bukan hanya template) dapat dilihat pada Kode 4.1.  
// Kode 4.1 — Contoh Entri Dataset RAFT Hasil Generasi  
 {  
   "instruction": "[Isi pertanyaan aktual dari dataset, contoh: Apa yang dimaksud dengan riba dalam ekonomi syariah?]",  
   "input": "Dokumen 1 (Oracle): [Isi teks chunk relevan dari dokumen OJK/KNEKS...]\n\nDokumen 2 (Distractor): [Isi teks chunk tidak relevan...]",  
   "output": "<Thought> Berdasarkan Dokumen 1, dijelaskan bahwa... [isi CoT aktual]. Oleh karena itu... </Thought>\nAnswer: [Jawaban akhir aktual]"  
 }  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsSfYxZo/kSGMYQLPJrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA4qrBdGuSdJuAAAAAElFTkSuQmCC)  
**4.4 Implementasi Fine-Tuning Model**  
Fine-tuning dilaksanakan menggunakan *framework* Unsloth pada model dasar Qwen-7B dengan metode QLoRA. Lingkungan komputasi yang digunakan adalah server **[nama server, misal: labkc-l40-debian]** dengan GPU  **[spesifikasi GPU, misal: NVIDIA L40 48GB VRAM]**.  
**4.4.1 Konfigurasi Hyperparameter Final**  
Berdasarkan serangkaian percobaan awal (*preliminary experiments*), konfigurasi hyperparameter final yang memberikan hasil terbaik adalah sebagaimana tercantum pada Tabel 4.5.  
**Tabel 4.5 — Konfigurasi Hyperparameter Final**  
| | | |  
|-|-|-|  
| **Parameter** | **Nilai** | **Keterangan** |   
| LoRA Rank (r) | [16 / 32] | Dipilih karena [alasan, misal: memberikan konvergensi lebih stabil] |   
| LoRA Alpha | [32 / 64] | — |   
| Learning Rate | [X] | — |   
| Batch Size | [X] | Disesuaikan dengan kapasitas VRAM |   
| Gradient Accumulation | [X] | Efektif mensimulasikan batch size [X] |   
| Quantization | 4-bit (NF4) | — |   
| Epoch | [X] | — |   
| Max Steps | [X] | — |   
| Total waktu pelatihan | [X] menit/jam | — |   
   
**4.4.2 Kurva Training Loss**  
Proses pelatihan dipantau secara *real-time* menggunakan Unsloth. Kurva  *training loss* menunjukkan penurunan yang konvergen dari nilai awal **[X]** hingga nilai akhir  **[X]**, sebagaimana ditunjukkan pada Gambar 4.1.  
***[Gambar 4.2 — Training Loss Curve]***  
 *  
 * *(Sisipkan grafik training loss dari hasil eksperimen di sini)*  
Pola konvergensi yang stabil mengindikasikan bahwa model berhasil mempelajari pola dataset RAFT tanpa mengalami *overfitting* yang signifikan. Hal ini didukung oleh nilai loss akhir yang berada pada rentang **[X]**, sesuai dengan target yang ditetapkan.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSPBCj7fFwtCmJHAjAU2QtIq6DIzW7UHAMBfnGt1V8fHEQAA3rsexOkF3va0dq8AAAAASUVORK5CYII=)  
**4.5 Hasil Konversi dan Kuantisasi Model (GGUF)**  
Setelah proses fine-tuning selesai, adapter LoRA digabungkan ke dalam *base model* dan selanjutnya dikonversi ke format GGUF untuk kebutuhan deployment menggunakan Ollama.  
**Tabel 4.6 — Perbandingan Ukuran Model Sebelum dan Sesudah Kuantisasi**  
| | | |  
|-|-|-|  
| **Format** | **Ukuran File** | **Persentase Kompresi** |   
| Model float16 (sebelum kuantisasi) | [X] GB | — |   
| GGUF q4_k_m | [X] GB | [X]% |   
| GGUF q8_0 | [X] GB | [X]% |   
| **Format yang dipilih** | **[q4_k_m / q8_0]** | **[X]%** |   
   
Metode **[q4_k_m / q8_0]** dipilih sebagai format deployment karena  **[alasan, misal: memberikan keseimbangan terbaik antara ukuran model dan kualitas output berdasarkan uji awal perbandingan keduanya]**.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OUQmAABBAsSeILQSjXgcrmkOs4J8IW4ItM7NXZwAA/MW1Vlt1fBwBAOC9+wEukwQ+V/SggAAAAABJRU5ErkJggg==)  
**4.6 Implementasi LLM Service API**  
Layanan API dikembangkan menggunakan Python dengan FastAPI sebagai *framework* utama dan Ollama sebagai  *inference engine* yang menjalankan model GGUF. Sistem ini menyediakan beberapa  *endpoint* RESTful yang dapat dikonsumsi oleh platform web maupun aplikasi mobile.  
**4.6.1 Daftar Endpoint API**  
**Tabel 4.7 — Daftar Endpoint API Chatbot**  
| | | |  
|-|-|-|  
| **Method** | **Endpoint** | **Deskripsi** |   
| POST | /api/v1/chat | Mengirim pesan dan menerima respons chatbot |   
| GET | /api/v1/history/{user_id} | Mengambil riwayat percakapan pengguna |   
| DELETE | /api/v1/history/{user_id} | Menghapus riwayat percakapan pengguna |   
| GET | /api/v1/health | Mengecek status layanan API |   
   
**4.6.2 Contoh Request dan Response**  
Kode 4.2 menunjukkan contoh *request* dan  *response* JSON pada endpoint utama /api/v1/chat.  
// Kode 4.2 — Contoh Request (POST /api/v1/chat)  
 {  
   "user_id": "user_001",  
   "message": "Apa perbedaan antara bank syariah dan bank konvensional?"  
 }  
   
 // Contoh Response (200 OK)  
 {  
   "status": "success",  
   "response": "[Contoh jawaban aktual dari model yang dikembangkan...]",  
   "latency_ms": [X],  
   "retrieved_docs": [  
     {  
       "source": "[nama dokumen sumber]",  
       "page": [X],  
       "relevance_score": [X]  
     }  
   ]  
 }  
   
**4.6.3 Hasil Pengujian Deployment**  
Pengujian menggunakan Postman menunjukkan seluruh endpoint merespons dengan status **200 OK** pada kondisi *single user*. Rata-rata latensi awal yang tercatat adalah **[X] ms**, dengan kecepatan generasi token sebesar  **[X] token/detik**.  
***[Gambar 4.3 — Screenshot Hasil Pengujian Postman]***  
 *  
 * *(Sisipkan screenshot Postman di sini)*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBACPykMH4NpGACyywEZJWQZeZ2aszAAD+4l6rrTo+jgAA8N71AL/CBEiG5xPoAAAAAElFTkSuQmCC)  
**4.7 Implementasi Integrasi Sistem Lintas Platform**  
Sistem chatbot yang dikembangkan diintegrasikan ke dalam ekosistem platform literasi ekonomi syariah yang terdiri dari tiga komponen:  
1. **Backend Chatbot API** (dikembangkan oleh penulis — Faturrohman, 2025): Sistem inferensi LLM berbasis FastAPI dan Ollama, dilengkapi dengan Firebase untuk autentikasi dan penyimpanan riwayat percakapan.  
2. **Aplikasi Mobile** (dikembangkan oleh Kamal, 2025): Antarmuka pengguna berbasis mobile yang mengakses layanan backend melalui HTTP.  
3. **Aplikasi Web** (dikembangkan oleh Susilo, 2025): Antarmuka berbasis web dengan fungsi yang ekuivalen.  
Seluruh komunikasi antar komponen dilakukan melalui protokol HTTP dengan format data JSON. Pengujian integrasi dilakukan dengan memverifikasi alur percakapan end-to-end dari input pengguna pada sisi frontend hingga respons yang dikembalikan oleh model LLM.  
***[Gambar 4.4 — Screenshot Tampilan Chatbot pada Aplikasi Mobile/Web]***  
 *  
 * *(Sisipkan screenshot tampilan antarmuka di sini)*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAALUlEQVR4nO3OQQ0AIAwEsAMlSJ0UrOFkGngRklZBR1WtJDsAAPzizNcDAADuNcKwAyU+nb+5AAAAAElFTkSuQmCC)  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3KsQ0AIRAEsUW6Qij1KvnevhMSYmKQ7GiCGd09k3wBAOAVf+2o4wYAwE1qAdYuAy151mgcAAAAAElFTkSuQmCC)  
**BAB 5 — EVALUASI DAN PEMBAHASAN**  
Bab ini menyajikan hasil evaluasi sistem yang telah dikembangkan berdasarkan tiga aspek utama yang menjawab rumusan masalah penelitian, yaitu: (1) kinerja teknis sistem, (2) kualitas jawaban model, dan (3) efektivitas edukasi chatbot dalam meningkatkan literasi ekonomi syariah. Selain itu, bab ini juga memuat pembahasan menyeluruh terhadap temuan-temuan yang diperoleh.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNhYMEBIpD4ArCJDyywEZJWQZeZOaorAAD+4l6rrTq/ngAA8Nr+AEqmA1hl45m5AAAAAElFTkSuQmCC)  
**5.1 Evaluasi Validitas dan Kepraktisan Chatbot (RM-1)**  
Evaluasi ini dilakukan untuk memastikan chatbot yang dikembangkan memenuhi standar validitas konten dan kepraktisan penggunaan sebelum diujicobakan secara lebih luas kepada siswa.  
**5.1.1 Uji Validitas**  
Validasi konten dilakukan oleh **[X] validator ahli** yang terdiri dari pakar di bidang ekonomi syariah dan teknologi pembelajaran. Instrumen validasi menggunakan lembar penilaian dengan skala Likert 1–5 yang mencakup aspek: kesesuaian materi, keakuratan informasi, kejelasan bahasa, dan kesesuaian dengan tingkat pemahaman siswa SMP–SMA.  
**Tabel 5.1 — Hasil Uji Validitas oleh Ahli**  
| | | | | | |  
|-|-|-|-|-|-|  
| **Aspek Penilaian** | **Validator 1** | **Validator 2** | **[Validator X]** | **Rata-rata** | **Kategori** |   
| Kesesuaian Materi | [X] | [X] | [X] | [X] | [Valid/Sangat Valid] |   
| Keakuratan Informasi | [X] | [X] | [X] | [X] | [Valid/Sangat Valid] |   
| Kejelasan Bahasa | [X] | [X] | [X] | [X] | [Valid/Sangat Valid] |   
| Kesesuaian Tingkat Siswa | [X] | [X] | [X] | [X] | [Valid/Sangat Valid] |   
| **Rata-rata Keseluruhan** |   |   |   | **[X]** | **[Valid/Sangat Valid]** |   
   
Hasil validasi menunjukkan nilai rata-rata keseluruhan sebesar **[X]** dari skala 5, yang termasuk dalam kategori  **[valid/sangat valid]**. Hal ini mengindikasikan bahwa konten yang dihasilkan oleh chatbot dinilai layak dan sesuai sebagai media pembelajaran ekonomi syariah.  
**5.1.2 Uji Kepraktisan**  
Uji kepraktisan dilakukan pada **[X] siswa** yang terdiri dari  **[X] siswa SMP** dan  **[X] siswa SMA** pada uji coba terbatas (*limited trial*). Aspek yang dinilai meliputi kemudahan penggunaan, kejelasan antarmuka, kecepatan respons, dan kepuasan pengguna secara umum.  
**Tabel 5.2 — Hasil Uji Kepraktisan oleh Siswa**  
| | | |  
|-|-|-|  
| **Aspek Kepraktisan** | **Rata-rata Skor** | **Kategori** |   
| Kemudahan Penggunaan | [X] | [Praktis/Sangat Praktis] |   
| Kejelasan Antarmuka | [X] | [Praktis/Sangat Praktis] |   
| Kecepatan Respons | [X] | [Praktis/Sangat Praktis] |   
| Kepuasan Pengguna Umum | [X] | [Praktis/Sangat Praktis] |   
| **Rata-rata Keseluruhan** | **[X]** | **[Praktis/Sangat Praktis]** |   
   
Skor kepraktisan rata-rata sebesar **[X]** menunjukkan bahwa chatbot yang dikembangkan mudah digunakan dan dapat diterima dengan baik oleh target pengguna. Beberapa catatan dari siswa terkait perbaikan antarmuka adalah sebagai berikut:  **[ringkasan masukan kualitatif siswa, jika ada]**.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsSfYxZo/jkUsYQLPJrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA4rDBc72meO5AAAAAElFTkSuQmCC)  
**5.2 Evaluasi Kualitas Jawaban Model (RM-3: Aspek Akurasi dan Relevansi)**  
Evaluasi kualitas jawaban dilakukan menggunakan pendekatan *hybrid*, yaitu kombinasi metrik berbasis referensi statistik (BLEU dan ROUGE) serta evaluasi berbasis model (RAGAS dengan LLM-as-a-Judge menggunakan DeepSeek).  
Dataset uji yang digunakan terdiri dari **[X] pasang pertanyaan-jawaban** yang diambil dari subset dataset RAFT yang tidak digunakan pada proses pelatihan (*held-out test set*). Untuk meminimalkan bias akibat sifat stokastik model, setiap pertanyaan diuji sebanyak **tiga kali iterasi** dan hasilnya dirata-ratakan.  
**5.2.1 Evaluasi Metrik BLEU dan ROUGE**  
**Tabel 5.3 — Hasil Evaluasi BLEU dan ROUGE**  
| | | | | | |  
|-|-|-|-|-|-|  
| **Model** | **BLEU-1** | **BLEU-4** | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |   
| Baseline (Qwen-7B tanpa fine-tuning) | [X] | [X] | [X] | [X] | [X] |   
| **Model RAFT (hasil fine-tuning)** | **[X]** | **[X]** | **[X]** | **[X]** | **[X]** |   
| Peningkatan (%) | +[X]% | +[X]% | +[X]% | +[X]% | +[X]% |   
   
Hasil evaluasi menunjukkan peningkatan yang signifikan pada seluruh metrik setelah fine-tuning dengan pendekatan RAFT. Peningkatan BLEU-4 sebesar **[X]%** mengindikasikan model yang dikembangkan lebih tepat secara leksikal dalam menggunakan terminologi ekonomi syariah dibandingkan model *baseline*. Sementara itu, peningkatan ROUGE-L sebesar **[X]%** menunjukkan bahwa model mampu mencakup lebih banyak informasi penting dari jawaban referensi.  
Perlu dicatat bahwa skor BLEU dan ROUGE yang tidak terlalu tinggi secara absolut merupakan hal yang wajar untuk task *open-ended question answering*, karena metrik ini sangat bergantung pada kemiripan leksikal dan tidak sepenuhnya menangkap kualitas semantik jawaban (Majumder et al., 2024). Oleh karena itu, evaluasi RAGAS digunakan sebagai pelengkap yang lebih representatif.  
**5.2.2 Evaluasi RAGAS**  
RAGAS (*Retrieval-Augmented Generation Assessment*) digunakan untuk menilai kualitas jawaban dari sisi semantik dan kesesuaian dengan konteks retrieval. Evaluasi dilakukan dengan menggunakan DeepSeek sebagai  *judge model*.  
**Tabel 5.4 — Hasil Evaluasi RAGAS**  
| | | | | | |  
|-|-|-|-|-|-|  
| **Metrik RAGAS** | **Iterasi 1** | **Iterasi 2** | **Iterasi 3** | **Rata-rata** | **Interpretasi** |   
| Faithfulness | [X] | [X] | [X] | **[X]** | [Baik/Cukup/Perlu Perbaikan] |   
| Answer Relevance | [X] | [X] | [X] | **[X]** | [Baik/Cukup/Perlu Perbaikan] |   
| Context Precision | [X] | [X] | [X] | **[X]** | [Baik/Cukup/Perlu Perbaikan] |   
   
Nilai *Faithfulness* sebesar **[X]** menunjukkan bahwa sekitar  **[X]%** pernyataan dalam jawaban model dapat ditelusuri dan didukung oleh konteks dokumen yang diberikan melalui mekanisme retrieval. Ini mengindikasikan tingkat halusinasi model yang  **[rendah/moderat]**, yang sejalan dengan tujuan penerapan metode RAFT untuk melatih model agar lebih bertumpu pada dokumen referensi (Zhang et al., 2024).  
Nilai *Answer Relevance* sebesar **[X]** menunjukkan bahwa jawaban yang dihasilkan secara semantik relevan dengan pertanyaan yang diajukan. Nilai *Context Precision* sebesar **[X]** menunjukkan kemampuan sistem retrieval dalam mengambil dokumen yang tepat dari basis pengetahuan.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OUQmAABBAsaeI2MKqV8RyJrGCfyJsCbbMzFldAQDwF/dWrdXx9QQAgNf2B/NkAzRb7P0YAAAAAElFTkSuQmCC)  
**5.3 Evaluasi Kinerja Teknis Sistem (RM-3: Aspek Performa)**  
Evaluasi kinerja teknis dilaksanakan melalui pengujian *load testing* menggunakan  *Locust* pada lingkungan jaringan lokal. Pengujian dilakukan dengan skema  *incremental load* pada beban 1, 5, dan 10 pengguna  *concurrent*, dengan dua skenario input: pertanyaan identik dan pertanyaan variatif. Setiap skenario beban diulang sebanyak **5 kali iterasi**.  
**Tabel 5.5 — Hasil Evaluasi Kinerja Teknis (Skenario Pertanyaan Variatif)**  
| | | | |  
|-|-|-|-|  
| **Jumlah Pengguna Concurrent** | **Rata-rata Latency (ms)** | **Throughput (token/s)** | **Error Rate (%)** |   
| 1 | [X] | [X] | [X]% |   
| 5 | [X] | [X] | [X]% |   
| 10 | [X] | [X] | [X]% |   
   
***[Gambar 5.1 — Grafik Latency vs. Jumlah Pengguna Concurrent]***  
 *  
 * *(Sisipkan grafik visualisasi hasil load testing di sini)*  
Hasil pengujian menunjukkan bahwa sistem mampu melayani hingga **[X] pengguna *****concurrent*** dengan latensi rata-rata di bawah **[X] ms** dan *error rate* **[X]%**, yang  **[memenuhi/mendekati]** target kualitas operasional yang ditetapkan pada Subbab 3.2.3.  
Peningkatan latensi yang terjadi seiring bertambahnya jumlah pengguna disebabkan oleh **[penjelasan teknis, misal: antrian pemrosesan pada inference engine Ollama yang bersifat sequential]**. Throughput rata-rata sebesar  **[X] token/detik** menunjukkan efisiensi komputasi model yang berjalan dalam format GGUF q4_k_m, konsisten dengan laporan efisiensi kuantisasi pada penelitian sebelumnya (Dettmers et al., 2023).  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUeCE4yeIiT9CRVMWGAjJK2CbjNzVGcAAPzF2qu7Wl9PAAB47XoA/vcF8exqpY4AAAAASUVORK5CYII=)  
**5.4 Evaluasi Efektivitas Edukasi (RM-2)**  
Evaluasi efektivitas edukasi dilakukan untuk mengukur dampak nyata chatbot terhadap peningkatan pemahaman literasi ekonomi syariah pada siswa. Metode yang digunakan adalah eksperimen *pre-test* dan  *post-test* dengan analisis  *Normalized Gain* (N-Gain).  
**5.4.1 Prosedur Evaluasi**  
Pengujian melibatkan total **60 siswa** yang terdiri dari  **30 siswa SMP [Nama SMP]** dan  **30 siswa SMA [Nama SMA]** di Surabaya. Alur evaluasi adalah sebagai berikut:  
1. Siswa mengerjakan *pre-test* (soal pilihan ganda/esai terkait ekonomi syariah, **[X] soal**)  
2. Siswa berinteraksi dengan chatbot selama **[X] menit/pertemuan**  
3. Siswa mengerjakan *post-test* dengan instrumen yang setara  
**5.4.2 Hasil Pre-test dan Post-test**  
**Tabel 5.6 — Rekapitulasi Hasil Pre-test dan Post-test**  
| | | | | | | |  
|-|-|-|-|-|-|-|  
| **Sekolah** | **N** | **Rata-rata Pre-test** | **Rata-rata Post-test** | **Selisih Skor** | **N-Gain (g)** | **Kategori** |   
| SMP [Nama] | 30 | [X] | [X] | +[X] | **[X]** | [Tinggi/Sedang/Rendah] |   
| SMA [Nama] | 30 | [X] | [X] | +[X] | **[X]** | [Tinggi/Sedang/Rendah] |   
| **Gabungan** | **60** | **[X]** | **[X]** | **+[X]** | **[X]** | **[Tinggi/Sedang/Rendah]** |   
   
***[Gambar 5.2 — Diagram Batang Perbandingan Pre-test dan Post-test per Kelompok]***  
Berdasarkan Tabel 5.6, nilai N-Gain pada kelompok SMP sebesar **[X]** masuk dalam kategori  **[Tinggi/Sedang/Rendah]** (g ≥ 0.7 / 0.3 ≤ g < 0.7 / g < 0.3), sedangkan pada kelompok SMA sebesar  **[X]** masuk dalam kategori  **[Tinggi/Sedang/Rendah]**. Secara keseluruhan, nilai N-Gain gabungan sebesar  **[X]** menunjukkan bahwa chatbot edukatif berbasis RAFT berhasil meningkatkan pemahaman literasi ekonomi syariah siswa secara  **[signifikan/moderat]**.  
Temuan ini sejalan dengan penelitian Guo dan Erdenebold (2025) yang melaporkan nilai rata-rata *perceived learning effectiveness* di atas 4.0 (skala Likert 5 poin) pada penggunaan chatbot AI dalam pendidikan, serta Shahzad et al. (2025) yang menunjukkan skor kepuasan dan pemahaman antara 3.8–4.3 pada aspek klarifikasi konsep.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNBCUpfD6ZYGZDAgAU2QtIq6DIzW7UHAMBfHGt1V+fXEwAAXrseHCoGAe/SKtAAAAAASUVORK5CYII=)  
**5.5 Pembahasan Keseluruhan**  
**5.5.1 Efektivitas Pendekatan RAFT**  
Hasil evaluasi secara keseluruhan mengonfirmasi bahwa pendekatan RAFT yang diterapkan dalam penelitian ini memberikan dampak positif yang terukur pada tiga dimensi yang dievaluasi. Dari sisi kualitas jawaban, peningkatan skor BLEU-4 sebesar **[X]%** dan nilai *Faithfulness* RAGAS sebesar **[X]** menunjukkan bahwa proses fine-tuning dengan konstruksi dataset yang menyertakan *distractor document* berhasil melatih model untuk lebih selektif dalam menggunakan informasi dari dokumen referensi, sebagaimana yang diargumentasikan Zhang et al. (2024).  
Dibandingkan dengan pendekatan RAG murni (*tanpa fine-tuning*), model RAFT yang dikembangkan menunjukkan keunggulan dalam hal **[temuan spesifik, misal: konsistensi jawaban dan penggunaan terminologi yang tepat]**. Hal ini mendukung argumen bahwa RAFT bukan sekadar melakukan *fine-tuning* biasa, melainkan juga melatih kemampuan penalaran model dalam konteks  *open-book* secara bersamaan (Zhang et al., 2024).  
**5.5.2 Keterbatasan Penelitian**  
Beberapa keterbatasan yang diidentifikasi dalam penelitian ini adalah sebagai berikut:  
1. **Cakupan dataset terbatas**: Dataset pelatihan dibatasi pada materi dasar ekonomi syariah dari OJK dan KNEKS. Materi yang lebih mendalam seperti fikih muamalah lanjutan dan perbedaan pendapat mazhab tidak tercakup, sehingga chatbot belum dapat menjawab pertanyaan di luar ruang lingkup ini.  
2. **Skala uji coba terbatas**: Uji efektivitas edukasi hanya melibatkan 60 siswa dari dua sekolah di Surabaya, sehingga generalisasi hasil ke populasi yang lebih luas memerlukan penelitian lanjutan dengan sampel yang lebih representatif.  
3. **Keterbatasan komputasi**: Penggunaan model 7B parameter dengan kuantisasi 4-bit, meskipun efisien, memiliki batas kemampuan dibandingkan model dengan parameter lebih besar. Kompresi model berpotensi menurunkan kualitas jawaban pada pertanyaan yang membutuhkan penalaran yang lebih kompleks.  
4. **Evaluasi latensi pada kondisi nyata**: Pengujian kinerja dilakukan pada lingkungan jaringan lokal, sehingga hasil latensi belum sepenuhnya merepresentasikan kondisi deployment di jaringan publik dengan variasi koneksi pengguna yang sesungguhnya.  
**5.5.3 Kontribusi Penelitian**  
Penelitian ini memberikan kontribusi dalam beberapa aspek:  
- Membuktikan bahwa pendekatan RAFT dapat diterapkan secara efektif pada domain literasi ekonomi syariah menggunakan model open-source skala menengah (7B parameter) dengan sumber daya komputasi yang terjangkau.  
- Menyediakan kerangka pengembangan media pembelajaran chatbot berbasis AI yang mengintegrasikan metodologi ADDIE dengan teknik fine-tuning modern, yang dapat dijadikan referensi untuk pengembangan sejenis di domain lain.  
- Menghasilkan dataset RAFT berbahasa Indonesia dalam domain ekonomi syariah sebagai sumber daya yang dapat dimanfaatkan oleh penelitian selanjutnya.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUfyVTCg9UygEBVsWGAjJK2CbjNzVGcAAPzFtapV7V9PAAB47X4AEWIEM8iQs0EAAAAASUVORK5CYII=)  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OUQmAABBAsSeILQSjXgcrmkOs4J8IW4ItM7NXZwAA/MW1Vlt1fBwBAOC9+wEukwQ+V/SggAAAAABJRU5ErkJggg==)  
**BAB 6 — PENUTUP**  
**6.1 Kesimpulan**  
Berdasarkan hasil penelitian dan evaluasi yang telah dilakukan, dapat ditarik kesimpulan sebagai berikut:  
1. **[Menjawab RM-1]** Chatbot edukatif berbasis LLM dengan pendekatan RAFT berhasil dikembangkan menggunakan kerangka kerja ADDIE. Sistem ini mencakup pipeline lengkap mulai dari pre-processing data, penyusunan dataset RAFT, fine-tuning model Qwen-7B, konversi GGUF, hingga deployment sebagai layanan API. Hasil uji validitas menunjukkan nilai rata-rata  **[X]** (kategori  **[valid/sangat valid]**) dan uji kepraktisan menunjukkan nilai  **[X]** (kategori  **[praktis/sangat praktis]**), mengindikasikan sistem yang dikembangkan layak digunakan sebagai media pembelajaran.  
2. **[Menjawab RM-2]** Chatbot yang dikembangkan terbukti efektif dalam meningkatkan pemahaman literasi ekonomi syariah siswa SMP dan SMA. Nilai N-Gain gabungan sebesar  **[X]** (kategori  **[Tinggi/Sedang]**) menunjukkan peningkatan pemahaman yang  **[signifikan/moderat]** setelah interaksi dengan chatbot, dibandingkan kondisi sebelum penggunaan.  
3. **[Menjawab RM-3]** Model LLM yang dikembangkan menunjukkan kinerja teknis yang memadai dengan latensi rata-rata  **[X] ms** pada beban 10 pengguna *concurrent* dan  *error rate* **[X]%**. Dari sisi kualitas jawaban, model mencapai nilai BLEU-4 sebesar  **[X]**, ROUGE-L sebesar  **[X]**, *Faithfulness* RAGAS sebesar **[X]**, dan *Answer Relevance* sebesar **[X]**, yang secara keseluruhan menunjukkan jawaban yang akurat, relevan, dan berbasis dokumen referensi yang valid.  
**6.2 Saran**  
Berdasarkan keterbatasan dan temuan penelitian ini, beberapa saran untuk penelitian dan pengembangan selanjutnya adalah:  
1. **Perluasan dataset**: Dataset pelatihan dapat diperluas mencakup materi ekonomi syariah yang lebih mendalam, seperti akad-akad muamalah spesifik, instrumen keuangan syariah kontemporer, dan konten berbasis fatwa DSN-MUI, guna meningkatkan cakupan dan kedalaman pengetahuan chatbot.  
2. **Eksplorasi model dasar yang lebih besar**: Penelitian lanjutan dapat mengeksplorasi penggunaan model dengan parameter lebih besar (misal: 13B atau 70B) yang dikuantisasi, untuk mengetahui trade-off antara kualitas jawaban dan kebutuhan komputasi.  
3. **Uji coba dengan sampel yang lebih luas**: Evaluasi efektivitas edukasi perlu diulang dengan sampel yang lebih besar dan beragam (mencakup berbagai daerah di Indonesia) untuk memvalidasi hasil penelitian secara lebih menyeluruh.  
4. **Pengembangan fitur adaptif**: Sistem dapat dikembangkan dengan fitur personalisasi adaptif, misalnya menyesuaikan tingkat kesulitan materi berdasarkan profil dan riwayat belajar siswa, untuk meningkatkan efektivitas pembelajaran lebih lanjut.  
5. **Evaluasi jangka panjang**: Penelitian longitudinal diperlukan untuk mengukur retensi pengetahuan dan dampak jangka panjang penggunaan chatbot terhadap literasi ekonomi syariah siswa.  
