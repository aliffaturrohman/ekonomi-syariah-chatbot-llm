import os
import json
import random
import time
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.utils.pydantic import BaseModel, Field

# =============================
# Konfigurasi
# =============================
INPUT_DIR = "../data/processed/02_cleaning_markdown_text_heading/"
OUTPUT_DIR = "../data/dataset_training_ver2/"
DB_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

MAX_QUESTIONS_PER_FILE = 200

# =============================
# Model LLM
# =============================
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.3, # Sedikit dinaikkan agar variasi soal lebih kreatif
    format="json",
    base_url="http://localhost:11434"
)

# =============================
# Data Structure
# =============================
class RaftData(BaseModel):
    question: str = Field(description="Pertanyaan yang menguji pemahaman konsep, bukan sekadar mencocokkan kata.")
    thought_process: str = Field(description="Analisis langkah demi langkah. Jelaskan BAGIAN MANA dari teks yang mendukung jawaban.")
    answer: str = Field(description="Jawaban akhir yang padat, jelas, dan akurat.")

parser = JsonOutputParser(pydantic_object=RaftData)

# =============================
# Prompt RAFT Versi PRO (Dipertajam)
# =============================
raft_system_prompt = """
Anda adalah Pakar Ekonomi Syariah yang sedang menyusun dataset untuk ujian sertifikasi.

Tugas Anda:
Diberikan sekumpulan potongan teks (dokumen), buatlah SATU pasang pertanyaan dan jawaban (QA) yang berkualitas tinggi.

Aturan Keras:
1. Pertanyaan harus bisa dijawab HANYA menggunakan informasi yang ada di dalam teks yang diberikan.
2. JANGAN membuat pertanyaan jika informasinya tidak lengkap atau ambigu.
3. 'thought_process' harus berisi penalaran logis: Identifikasi mana dokumen yang relevan (Oracle) dan mana yang tidak relevan (Distractor), lalu kutip faktanya.
4. Jawaban tidak boleh bertele-tele. Langsung pada inti persoalan.

CONTOH OUTPUT YANG DIHARAPKAN (FEW-SHOT):

--- Contoh 1 ---
Input Dokumen:
[Dokumen A]: Riba fadhl adalah tukar menukar dua barang yang sejenis dengan takaran yang berbeda...
[Dokumen B]: Sejarah masuknya Islam ke Indonesia dimulai pada abad ke-7 masehi melalui jalur perdagangan...

Output JSON:
{{
    "question": "Apa yang dimaksud dengan Riba Fadhl berdasarkan referensi?",
    "thought_process": "Dokumen B membahas sejarah masuknya Islam, tidak relevan dengan pertanyaan fikih muamalah. Dokumen A secara spesifik mendefinisikan Riba Fadhl sebagai pertukaran barang sejenis dengan takaran berbeda. Saya akan menyusun jawaban berdasarkan Dokumen A.",
    "answer": "Riba Fadhl adalah jenis riba yang terjadi dalam pertukaran antarbarang yang sejenis namun dengan takaran atau kadar yang berbeda."
}}

--- Contoh 2 ---
Input Dokumen:
[Dokumen A]: Akad Murabahah adalah akad jual beli barang dengan menyatakan harga perolehan dan keuntungan (margin) yang disepakati...
[Dokumen B]: Syarat sah shalat ada lima, yaitu suci dari hadas, menutup aurat...

Output JSON:
{{
    "question": "Jelaskan karakteristik utama dari akad Murabahah!",
    "thought_process": "Dokumen B menjelaskan syarat shalat (Fikih Ibadah), jadi ini adalah distractor. Dokumen A relevan karena mendefinisikan akad Murabahah. Poin kuncinya adalah 'menyatakan harga perolehan' dan 'keuntungan/margin'.",
    "answer": "Karakteristik utama akad Murabahah adalah adanya transparansi harga perolehan barang dan kesepakatan mengenai jumlah keuntungan (margin) antara penjual dan pembeli."
}}

Format Output Wajib JSON:
{format_instructions}
"""

raft_prompt = ChatPromptTemplate.from_messages([
    ("system", raft_system_prompt),
    ("user", "Kumpulan Dokumen Referensi:\n{context_docs}\n\nBuat 1 soal RAFT berdasarkan referensi di atas.")
])

# =============================
# Distractor Selection
# =============================
def get_distractors(vectorstore, oracle_doc, all_chunks, min_score=0.35, k=10):
    # Cari dokumen yang mirip tapi bukan dokumen yang sama
    results = vectorstore.similarity_search_with_score(oracle_doc, k=k)

    distractor_pool = [
        doc.page_content for doc, score in results
        if score > min_score and doc.page_content != oracle_doc
    ]

    # Jika kurang, ambil random dari file yang sama
    if len(distractor_pool) < 3:
        shortage = 3 - len(distractor_pool)
        candidates = [c for c in all_chunks if c != oracle_doc]
        if len(candidates) >= shortage:
            distractor_pool += random.sample(candidates, shortage)
        else:
            distractor_pool += candidates

    return distractor_pool[:3] # Ambil 3 distractor teratas

# =============================
# Quality Filter
# =============================
def quality_filter(result):
    q = result.get("question", "").lower()
    a = result.get("answer", "").lower()
    t = result.get("thought_process", "").lower()

    if len(q) < 20: return False # Pertanyaan terlalu pendek
    if "paragraf" in q or "teks di atas" in q: return False # Hindari referensi meta
    if not t: return False # Thought process kosong
    if len(a.split()) < 5: return False # Jawaban terlalu pendek
    
    return True

# =============================
# Process Single File
# =============================
def process_single_file(file_path, filename, vectorstore, chain):
    print(f"\n📂 Memproses: {filename}")

    clean_name = filename.replace(".md", "")
    output_path = os.path.join(OUTPUT_DIR, f"raft_dataset_{clean_name}.jsonl")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = text_splitter.split_text(raw_text)

    # Sampling jika chunks terlalu banyak
    if MAX_QUESTIONS_PER_FILE < len(chunks):
        target_chunks = random.sample(chunks, MAX_QUESTIONS_PER_FILE)
    else:
        target_chunks = chunks

    success = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for oracle_doc in tqdm(target_chunks, desc=f"Gen {clean_name}", leave=False):
            # 1. Ambil Distractors
            distractors = get_distractors(vectorstore, oracle_doc, chunks)
            
            # 2. SHUFFLING (PENTING! Agar posisi jawaban benar tidak selalu di atas)
            context_list = [oracle_doc] + distractors
            random.shuffle(context_list) # Acak urutan dokumen
            
            final_context_str = "\n\n---\n\n".join(context_list)

            try:
                # 3. Generate Soal via LLM
                response = chain.invoke({
                    "context_docs": final_context_str,
                    "format_instructions": parser.get_format_instructions()
                })

                if not quality_filter(response):
                    continue

                # 4. FORMATTING DATASET (PERBAIKAN UTAMA DI SINI)
                # Kita bungkus thought_process dengan tag XML agar sesuai dengan kode UI Streamlit
                formatted_thought = f"<thought>\n{response['thought_process']}\n</thought>"
                final_answer = f"{formatted_thought}\n\n{response['answer']}"

                training_entry = {
                    "conversations": [
                        {
                            "from": "system",
                            "value": "Anda adalah HANIF, asisten AI Ekonomi Syariah. Jawablah pertanyaan pengguna dengan akurat berdasarkan konteks yang diberikan. Mulailah dengan analisis mendalam menggunakan tag <thought>."
                        },
                        {
                            "from": "human",
                            "value": f"Konteks:\n{final_context_str}\n\nPertanyaan: {response['question']}"
                        },
                        {
                            "from": "gpt",
                            "value": final_answer # SUDAH TERMASUK TAG <thought>
                        }
                    ]
                }

                out.write(json.dumps(training_entry, ensure_ascii=False) + "\n")
                out.flush()

                success += 1
                time.sleep(0.05) # Sedikit delay agar GPU adem

            except Exception as e:
                # print(f"Error: {e}") # Uncomment jika ingin debug
                continue

    return success

# =============================
# Chunking
# =============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Diperbesar sedikit agar konteks lebih utuh
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n\n", "\n"]
)

# =============================
# Main Pipeline
# =============================
def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Folder input tidak ditemukan: {INPUT_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("⏳ Memuat Embedding & Vector Store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    chain = raft_prompt | llm | parser

    file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(".md")]
    print(f"📄 Total file: {len(file_list)}")

    total = 0
    for filename in file_list:
        count = process_single_file(os.path.join(INPUT_DIR, filename), filename, vectorstore, chain)
        total += count
        print(f"   ✓ {filename} → {count} soal")

    print(f"\n🚀 TOTAL SELESAI: {total} data generated.")
    print(f"📁 Siap untuk training! Cek folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()