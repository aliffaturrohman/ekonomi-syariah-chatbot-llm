

import os
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm # Progress bar

INPUT_MD_DIR = "../data/processed/01_markdown_converted_from_pdf"
OUTPUT_DIR = "../data/processed/02B_cleaning_markdown_llm"

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=1,
    base_url="http://localhost:11434"
)

system_prompt = """
Kamu adalah asisten editor dokumen profesional. Tugasmu adalah memperbaiki format Markdown dengan sangat hati-hati tanpa mengubah makna atau isi teks.

Instruksi Utama:
1. Perbaiki struktur Heading (#, ##, ###, ####).
   - HANYA jadikan heading jika teks tersebut *jelas* merupakan Judul Bab, Sub-Bab, atau Subjudul resmi seperti “Pendahuluan”, “Kesimpulan”, “Contoh”, dll.
   - Jika sebuah kalimat panjang atau paragraf biasa diawali tanda '#', hapus tanda '#'.
   - Jika ada teks seperti “terjemahan” yang diawali '#', hapus tanda '#' tersebut.
   - JANGAN membuat heading baru jika tidak yakin 100%.

2. Jangan menambah, merangkum, atau menghilangkan konten apa pun.
   - Tidak boleh memotong ayat, hadits, kutipan, atau paragraf.
   - Tidak boleh menambahkan penjelasan baru.

3. Format ayat atau hadits:
   - Jika mendeteksi ayat/hadits dan terjemahannya, ubah menjadi format berikut:
     ```plain teks
         [ayat atau hadits]
         
         terjemahan:
         [teks terjemahan][ayat alquran/hadits]
     ```
     atau kalau tidak ada huruf arabnya, konteks berikut:
    ```plain teks        
        terjemahan:
        [teks terjemahan][ayat alquran/hadits]
    ```
   - Jangan menambahkan tanda '#' pada bagian ini.

4. Perbaiki list agar valid Markdown:
   - Jika input tampak seperti daftar namun rusak (mis. “1 ) ...” atau “-… tanpa spasi”), rapikan menjadi:
     - List bernomor: `1.`, `2.`, `3.` sesuai urutan input.
     - List bullet: `-` atau `*` sesuai format asli.
   - JANGAN mengubah urutan list.
   - JANGAN menambah item baru.
   - JANGAN menggabungkan dua list menjadi satu.

5. Rapikan spasi, pemisahan paragraf, dan baris yang terpotong.
   - Satukan baris terputus yang seharusnya satu paragraf.
   - Hindari double spacing, kecuali memang diperlukan untuk Markdown.

6. Jangan mengubah kata atau isi kecuali:
   - Jelas merupakan typo (contoh: “pekeria” -> “pekerja”).
   - Perbaikan dilakukan tanpa mengubah makna dan tidak menyentuh ayat/hadits.

7. Output-kan hanya teks Markdown yang sudah diperbaiki.
   - Tidak perlu ```markdown.
   - Tidak perlu komentar penjelasan.
   
8. Apabila terdapat elemen seperti <sup>1</sup> berada didepan teks, mohon 1 paragraf dihapus karena itu tidak termasuk ke dalam materi inti pembahasan (hanya referensi saja).
9. Tidak perlu menggunakan --- atau garis pemisah lainnya di dalam teks.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Perbaiki teks Markdown berikut:\n\n{text}")
])

chain = prompt_template | llm | StrOutputParser()

def clean_md(INPUT_MD_PATH, OUTPUT_DIR):
    if not os.path.exists(INPUT_MD_PATH):
        print(f"File tidak ditemukan: {INPUT_MD_PATH}")
        return

    print(f"Membaca file: {INPUT_MD_PATH}")
    with open(INPUT_MD_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 1. Pecah teks jadi chunk kecil
    print("Memecah dokumen untuk diproses per bagian...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=250
    )
    chunks = text_splitter.split_text(raw_text)

    cleaned_chunks = []
    
    print(f"Memulai pembersihan menggunakan LLM ({len(chunks)} bagian)...")
    
    for i, chunk in enumerate(tqdm(chunks)):
        try:
            response = chain.invoke({"text": chunk})
            cleaned_chunks.append(response)
        except Exception as e:
            print(f"Error pada chunk {i}: {e}")
            cleaned_chunks.append(chunk)

    # 2. Gabungkan kembali
    full_cleaned_text = "\n\n".join(cleaned_chunks)

    # 3. Simpan
    print(f"Menyimpan hasil bersih ke: {OUTPUT_DIR}")
    with open(OUTPUT_DIR, "w", encoding="utf-8") as f:
        f.write(full_cleaned_text)

    print("Selesai!")

def main():
    for filename in os.listdir(INPUT_MD_DIR):
        if filename.endswith(".md"):
            input_md_path = os.path.join(INPUT_MD_DIR, filename)
            output_md_path = os.path.join(OUTPUT_DIR, filename)
            clean_md(input_md_path, output_md_path)

if __name__ == "__main__":
    main()