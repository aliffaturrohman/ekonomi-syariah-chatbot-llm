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

INPUT_DIR = "../data/processed/02_cleaning_markdown_text_heading/"
OUTPUT_DIR = "../data/dataset_training/"
DB_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Batasi jumlah soal PER FILE (Misal: 10 soal per Bab)
MAX_QUESTIONS_PER_FILE = 200

# --- SETUP OLLAMA ---
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.1, 
    format="json",
    base_url="http://localhost:11434"
)

class RaftData(BaseModel):
    question: str = Field(description="Pertanyaan logis berdasarkan dokumen benar")
    thought_process: str = Field(description="Analisis kenapa dokumen benar dipilih dan dokumen lain salah")
    answer: str = Field(description="Jawaban akhir yang akurat")

parser = JsonOutputParser(pydantic_object=RaftData)

raft_system_prompt = """
PERAN:
Anda adalah Dosen Ekonomi Syariah yang sedang membuat bank soal ujian.
Anda diberikan dua jenis dokumen:
1. ORACLE (Sumber Kebenaran)
2. DISTRACTOR (Dokumen Pengalih/Salah)

TUGAS:
Buatlah satu pasang Pertanyaan dan Jawaban berdasarkan ORACLE.

ATURAN WAJIB (NEGATIVE CONSTRAINTS):
1. DILARANG KERAS menyebut kata "Oracle", "Distractor", "Dokumen 1", "Dokumen Benar", atau "Teks diatas" di dalam kolom 'question' dan 'answer'.
2. Pertanyaan harus terdengar NATURAL, seolah-olah user tidak tahu menahu soal dokumen yang anda pegang.
3. Jawaban harus langsung ke inti materi.

FORMAT OUTPUT JSON:
{{
    "question": "Pertanyaan natural...",
    "thought_process": "Analisis singkat...",
    "answer": "Jawaban materi langsung..."
}}

{format_instructions}
"""

raft_prompt = ChatPromptTemplate.from_messages([
    ("system", raft_system_prompt),
    ("human", "ORACLE:\n{oracle}\n\nDISTRACTORS:\n{distractors}\n\nBuat 1 pertanyaan, thought process, dan jawaban.")
])

# --- FUNGSI GENERATE PER FILE ---
def process_single_file(file_path, filename, vectorstore, chain):
    """
    Fungsi ini memproses 1 file MD dan menghasilkan 1 file JSONL
    """

    clean_name = filename.replace(".md", "")
    output_filename = f"raft_dataset_{clean_name}.jsonl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"\n📂 Memproses: {filename}")
    print(f"   Target Output: {output_filename}")

    # 2. Baca & Chunking File Ini
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(raw_text)
    
    # 3. Sampling (Batasi jumlah soal per file)
    if MAX_QUESTIONS_PER_FILE and MAX_QUESTIONS_PER_FILE < len(chunks):
        target_chunks = random.sample(chunks, MAX_QUESTIONS_PER_FILE)
    else:
        target_chunks = chunks

    # 4. Generate & Write
    success_count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        
        # Gunakan tqdm khusus untuk file ini
        for i, oracle_doc in enumerate(tqdm(target_chunks, desc=f"Gen {clean_name}", leave=False)):
            try:
                # A. Cari Distractor (Global dari DB)
                results = vectorstore.similarity_search(oracle_doc, k=4)
                distractor_docs = [doc.page_content for doc in results if doc.page_content != oracle_doc][:3]

                if len(distractor_docs) < 1: continue

                # B. Shuffle RAFT
                all_docs = [oracle_doc] + distractor_docs
                random.shuffle(all_docs)

                # C. Generate LLM
                response = chain.invoke({
                    "oracle": oracle_doc,
                    "distractors": "\n---\n".join(distractor_docs),
                    "format_instructions": parser.get_format_instructions()
                })

                # D. Format Training
                final_context_str = "\n\n---\n\n".join(all_docs)
                training_entry = {
                    "conversations": [
                        {
                            "from": "system",
                            "value": "Kamu adalah asisten AI yang ahli dalam Ekonomi Syariah. Gunakan konteks yang diberikan untuk menjawab pertanyaan dengan akurat dan lengkap. Berikan analisis sebelum menjawab."
                        },
                        {
                            "from": "human",
                            "value": f"Konteks:\n{final_context_str}\n\nPertanyaan: {response['question']}"
                        },
                        {
                            "from": "gpt",
                            "value": f"{response['thought_process']}\n\n{response['answer']}"
                        }
                    ]
                }

                f_out.write(json.dumps(training_entry, ensure_ascii=False) + "\n")
                f_out.flush()
                success_count += 1
                time.sleep(0.1)

            except Exception:
                continue
    
    return success_count

# --- FUNGSI UTAMA ---
def main():
    # 1. Setup Folder
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Folder input tidak ditemukan: {INPUT_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Setup Resources (Sekali saja di awal)
    print("⚙️  Loading Model Embedding & ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    chain = raft_prompt | llm | parser

    # 3. Loop Semua File Markdown
    file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(".md")]
    print(f"📄 Ditemukan {len(file_list)} file Markdown untuk diproses.")

    total_soal_dibuat = 0

    for filename in file_list:
        file_path = os.path.join(INPUT_DIR, filename)
        
        # Panggil fungsi proses per file
        count = process_single_file(file_path, filename, vectorstore, chain)
        total_soal_dibuat += count
        
        print(f"   ✅ Selesai: {filename} -> {count} soal.")

    print(f"\n🚀 SELESAI TOTAL! Berhasil membuat {total_soal_dibuat} data latih.")
    print(f"📁 Cek folder output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()