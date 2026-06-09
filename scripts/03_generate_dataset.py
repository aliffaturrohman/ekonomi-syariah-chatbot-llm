import os
import json
import random
import time
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.utils.pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================
# Konfigurasi
# =============================
INPUT_DIR = "data/processed/markdown_converted_from_pdf"
OUTPUT_DIR = "data/dataset_training_ver2/"
DB_DIR = "vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

MAX_QUESTIONS_PER_FILE = 500

# =============================
# Model LLM (OpenRouter)
# =============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

# Model IDs
FREE_MODEL = "deepseek/deepseek-v4-flash:free"
PAID_MODEL = "deepseek/deepseek-v4-flash"

def get_llm(model_name):
    return ChatOpenAI(
        model=model_name,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=BASE_URL,
        temperature=0.3,
        model_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://github.com/alif-faturrohman/ekonomi-syariah-chatbot-llm",
                "X-Title": "HANIF Dataset Generator"
            }
        }
    )

# Start with free model
current_model = FREE_MODEL
llm = get_llm(current_model)
# Chain akan diinisialisasi setelah prompt dan parser didefinisikan

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
Anda adalah HANIF (Helpful AI for Noble Islamic Finance), seorang Pakar Ekonomi Syariah senior yang bertugas menyusun materi edukasi berkualitas tinggi.

Tugas Anda:
Berdasarkan dokumen referensi yang diberikan, buatlah SATU pasang pertanyaan dan jawaban (QA) yang mendalam.

Gaya Bahasa & Tone:
1. EDUKATIF & MENJELASKAN: Jangan menjawab seperti kamus yang kaku. Jawablah seperti seorang guru yang sedang menjelaskan konsep kepada muridnya dengan bahasa yang sopan, jelas, dan mudah dipahami.
2. STRUKTUR: Jika diperlukan, gunakan poin-poin atau langkah-langkah dalam penjelasan agar informasi lebih teratur.
3. PROFESIONAL: Gunakan terminologi Ekonomi Syariah yang tepat.

Aturan Keras:
1. Pertanyaan harus bisa dijawab HANYA menggunakan informasi yang ada di dalam teks yang diberikan.
2. 'thought_process' harus berisi penalaran logis: Identifikasi mana dokumen yang relevan (Oracle) dan kutip faktanya, serta jelaskan kenapa dokumen lain diabaikan.
3. DILARANG KERAS menggunakan frasa pembuka seperti "Menurut teks...", "Berdasarkan dokumen...", atau "Teks menyebutkan...". Langsung berikan penjelasan ahli Anda.
4. JANGAN mereferensikan gambar, tabel, atau nomor halaman (misal: "lihat gambar 1", "pada halaman 20").

CONTOH OUTPUT YANG DIHARAPKAN:
{{
    "question": "Mengapa prinsip keadilan sangat krusial dalam akad perbankan syariah?",
    "thought_process": "Dokumen A menjelaskan tentang etika bisnis umum, sementara Dokumen B secara spesifik membahas pilar keadilan dalam perbankan syariah sebagai antitesis dari sistem bunga. Saya akan menggunakan Dokumen B.",
    "answer": "Prinsip keadilan merupakan fondasi utama dalam perbankan syariah karena sistem ini beroperasi atas dasar pembagian risiko dan keuntungan (risk and profit sharing). Berbeda dengan sistem konvensional, keadilan di sini memastikan bahwa tidak ada satu pihak pun yang mengeksploitasi pihak lain; jika terjadi keuntungan maka dinikmati bersama, dan jika terjadi risiko maka ditanggung sesuai porsi kesepakatan akad."
}}

Format Output Wajib JSON:
{format_instructions}
"""

raft_prompt = ChatPromptTemplate.from_messages([
    ("system", raft_system_prompt),
    ("user", "Kumpulan Dokumen Referensi:\n{context_docs}\n\nBuat 1 soal RAFT berdasarkan referensi di atas.")
])

# Initialize global chain after prompt and parser are ready
chain = raft_prompt | llm | parser

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
def process_single_file(file_path, filename, vectorstore):
    global llm, current_model, chain
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
                try:
                    response = chain.invoke({
                        "context_docs": final_context_str,
                        "format_instructions": parser.get_format_instructions()
                    })
                except Exception as e:
                    # Fallback to paid model if free model fails
                    if current_model == FREE_MODEL:
                        print(f"\n⚠️ Free model exhausted or error: {e}. Switching to PAID model...")
                        current_model = PAID_MODEL
                        llm = get_llm(current_model)
                        # Rebuild chain with new llm
                        chain = raft_prompt | llm | parser
                        # Retry once
                        response = chain.invoke({
                            "context_docs": final_context_str,
                            "format_instructions": parser.get_format_instructions()
                        })
                    else:
                        raise e

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

    print("⏳ Membuat Chain LLM...")
    # Chain sudah didefinisikan secara global

    file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(".md")]
    print(f"📄 Total file: {len(file_list)}")

    total = 0
    for filename in file_list:
        count = process_single_file(os.path.join(INPUT_DIR, filename), filename, vectorstore)
        total += count
        print(f"   ✓ {filename} → {count} soal")

    print(f"\n🚀 TOTAL SELESAI: {total} data generated.")
    print(f"📁 Siap untuk training! Cek folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()