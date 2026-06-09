import os
import json
import time
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.utils.pydantic import BaseModel, Field

# Load environment variables dari file .env
load_dotenv()

# --- KONFIGURASI PATH DINAMIS ---
# Memastikan script bisa dipanggil dari folder mana pun dengan aman
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUT_SINGLE_DIR = os.path.join(DATA_DIR, "dataset_training_ver2")
OUTPUT_SINGLE_DIR = os.path.join(DATA_DIR, "dataset_training_ver2_augmented")
MASTER_FILE_PATH = os.path.join(DATA_DIR, "MASTER_RAFT_DATASET.jsonl")
MASTER_OUTPUT_PATH = os.path.join(DATA_DIR, "MASTER_RAFT_DATASET_AUGMENTED.jsonl")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

# --- MODEL DEEPSEEK PILIHAN ---
MODEL_OPTIONS = {
    "1": ("deepseek/deepseek-chat", "DeepSeek-V3 (Flagship Chat - Cepat, Cerdas & Hemat)"),
    "2": ("deepseek/deepseek-r1", "DeepSeek-R1 (Flagship Reasoning - Sangat cocok untuk thought process mendalam)"),
    "3": ("deepseek/deepseek-r1-distill-llama-70b", "DeepSeek-R1 Distill Llama 70B (Keseimbangan performa & biaya)"),
    "4": ("deepseek/deepseek-chat:free", "DeepSeek-V3 Free (Jika tersedia di OpenRouter)"),
}

# =============================
# Struktur Data JSON Output
# =============================
class RaftData(BaseModel):
    question: str = Field(description="Pertanyaan Ekonomi Syariah yang menantang, natural, dan mendalam hasil augmentasi.")
    thought_process: str = Field(description="Proses berpikir langkah-demi-langkah yang mendalam, analitis, dan melacak dokumen referensi.")
    answer: str = Field(description="Jawaban akhir yang komprehensif, edukatif, dan ramah/sopan khas HANIF.")

parser = JsonOutputParser(pydantic_object=RaftData)

# =============================
# Prompt Augmentasi RAFT
# =============================
raft_augment_system_prompt = """
Anda adalah HANIF (Helpful AI for Noble Islamic Finance), seorang Pakar Ekonomi Syariah senior dan asisten editor AI profesional.
Tugas Anda adalah **mengaugmentasi (meningkatkan kualitas, memperluas, dan memoles)** sepasang Pertanyaan (Question), Proses Berpikir (Thought Process), dan Jawaban (Answer) berdasarkan Dokumen Referensi yang disediakan.

Tujuan utama augmentasi:
1. **Pertanyaan (Question)**: Buat pertanyaan menjadi lebih natural, menantang (menguji konsep mendalam), variatif, serta memiliki kualitas literasi tinggi. Pastikan pertanyaan HANYA dapat dijawab berdasarkan Dokumen Referensi yang diberikan.
2. **Proses Berpikir (Thought Process)**: Jelaskan secara terperinci langkah-langkah logika pemikiran. Identifikasi dokumen/paragraf mana yang relevan (Oracle), kutip fakta penting, dan sebutkan alasan mengapa dokumen pendukung lain diabaikan atau dianggap sebagai distraksi.
3. **Jawaban (Answer)**: Berikan jawaban yang komprehensif, edukatif, dan sopan dalam persona HANIF. Gunakan struktur yang rapi (seperti poin-poin atau langkah-langkah jika relevan) agar mudah dipahami siswa atau praktisi. Pastikan terminologi Ekonomi Syariah digunakan secara tepat.

Aturan Keras:
1. DILARANG menggunakan kata pembuka klise seperti "Berdasarkan dokumen...", "Menurut teks...", "Teks menyatakan...". Langsung berikan penjelasan ahli Anda.
2. JANGAN mereferensikan elemen meta dokumen seperti nomor halaman, nomor gambar, atau tabel (misalnya "lihat gambar 1", "pada halaman 20").
3. Isi dan substansi jawaban harus tetap 100% akurat dengan Dokumen Referensi. Jangan mengarang fakta di luar dokumen!

Berikut adalah input yang harus Anda tingkatkan:
- Dokumen Referensi:
{context_docs}

- Pertanyaan Asli: {original_question}
- Proses Berpikir Asli: {original_thought}
- Jawaban Asli: {original_answer}

Silakan hasilkan versi augmentasi yang jauh lebih berkualitas dalam format JSON.
Format Output Wajib JSON:
{format_instructions}
"""

# =============================
# Inisialisasi LLM
# =============================
def get_llm(model_name):
    return ChatOpenAI(
        model=model_name,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=BASE_URL,
        temperature=0.3,
        model_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://github.com/alif-faturrohman/ekonomi-syariah-chatbot-llm",
                "X-Title": "HANIF Dataset Augmenter"
            }
        }
    )

# =============================
# Parser Dataset ShareGPT
# =============================
def parse_training_entry(entry):
    """
    Parses a single ShareGPT training entry to extract:
    - context (original documents)
    - original question
    - original thought process
    - original answer
    """
    conversations = entry.get("conversations", [])
    if len(conversations) < 3:
        return None
    
    # Parse human prompt (Context + Question)
    human_val = conversations[1].get("value", "")
    
    context = ""
    question = ""
    
    # Mencari pemisah "Pertanyaan:"
    if "Pertanyaan:" in human_val:
        parts = human_val.split("Pertanyaan:")
        context_part = parts[0]
        question = parts[1].strip()
        
        if "Konteks:" in context_part:
            context = context_part.split("Konteks:")[1].strip()
        else:
            context = context_part.strip()
    else:
        # Fallback jika tidak ada kata kunci "Pertanyaan:"
        context = human_val
        question = ""
        
    # Parse GPT response (Thought + Answer)
    gpt_val = conversations[2].get("value", "")
    
    thought = ""
    answer = ""
    
    if "<thought>" in gpt_val and "</thought>" in gpt_val:
        parts = gpt_val.split("</thought>")
        thought_part = parts[0]
        answer = parts[1].strip()
        
        if "<thought>" in thought_part:
            thought = thought_part.split("<thought>")[1].strip()
        else:
            thought = thought_part.strip()
    else:
        # Fallback jika tidak ada tag <thought>
        thought = ""
        answer = gpt_val.strip()
        
    return {
        "context": context,
        "original_question": question,
        "original_thought": thought,
        "original_answer": answer
    }

# =============================
# Logika Retry & Filter Kualitas
# =============================
def extract_json_from_text(text):
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            last_triple = text.rfind("```")
            if last_triple > first_newline:
                text = text[first_newline:last_triple].strip()
            else:
                text = text[first_newline:].strip()
    return text

def clean_invalid_escapes(json_str):
    import re
    def replace_escape(match):
        full_match = match.group(0)
        char = match.group(1)
        if char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't']:
            return full_match
        if char.startswith('u'):
            if re.match(r'^\\u[0-9a-fA-F]{4}', full_match):
                return full_match
        return char

    # Match backslash followed by any character or u-escape
    return re.sub(r'\\(u[0-9a-fA-F]{0,4}|.)', replace_escape, json_str)

def call_llm_with_retry(chain, inputs, max_retries=5, initial_delay=2):
    import json
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            raw_response = chain.invoke(inputs)
            response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            # Clean and parse JSON
            json_str = extract_json_from_text(response_text)
            json_str_cleaned = clean_invalid_escapes(json_str)
            response = json.loads(json_str_cleaned)
            
            # Verify keys
            if not isinstance(response, dict) or not all(k in response for k in ["question", "thought_process", "answer"]):
                raise ValueError("JSON response does not contain required fields ('question', 'thought_process', 'answer')")
                
            return response
        except Exception as e:
            # Jika limit token terlampaui atau kendala server
            print(f"\n⚠️ Percobaan {attempt + 1}/{max_retries} gagal: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2

def quality_filter(result):
    q = result.get("question", "").lower()
    a = result.get("answer", "").lower()
    t = result.get("thought_process", "").lower()

    if len(q) < 20: return False  # Pertanyaan terlalu pendek
    if "paragraf" in q or "teks di atas" in q: return False  # Hindari referensi meta
    if not t: return False  # Thought process kosong
    if len(a.split()) < 5: return False  # Jawaban terlalu pendek
    
    return True

# =============================
# Fitur Resume (Skip yang sudah diproses)
# =============================
def get_already_processed_keys(output_path):
    processed_keys = set()
    if not os.path.exists(output_path):
        return processed_keys
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    parsed = parse_training_entry(entry)
                    if parsed and parsed.get("context"):
                        # Normalize context key
                        key = " ".join(parsed["context"].split())
                        processed_keys.add(key)
    except Exception as e:
        print(f"⚠️ Gagal membaca file output lama untuk fitur resume: {e}")
    return processed_keys

# =============================
# Proses File Dataset Tunggal
# =============================
def process_file(input_path, output_path, chain, limit=None):
    if not os.path.exists(input_path):
        print(f"❌ File input tidak ditemukan: {input_path}")
        return 0
    
    # Buat directory output jika belum ada
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Membaca seluruh data input dengan index baris asli (1-based)
    input_data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    input_data.append((idx + 1, json.loads(line)))
    except Exception as e:
        print(f"❌ Gagal membaca file input {input_path}: {e}")
        return 0
        
    total_samples = len(input_data)
    if total_samples == 0:
        print(f"⚠️ File input kosong: {input_path}")
        return 0
        
    # Fitur resume berbasis set
    processed_keys = get_already_processed_keys(output_path)
    
    # Saring data yang belum diproses berdasarkan kecocokan konteks
    remaining_data = []
    skipped_count = 0
    for line_num, entry in input_data:
        parsed = parse_training_entry(entry)
        if parsed and parsed.get("context"):
            key = " ".join(parsed["context"].split())
            if key in processed_keys:
                skipped_count += 1
                continue
        remaining_data.append((line_num, entry))
        
    if skipped_count >= total_samples:
        print(f"✅ File {os.path.basename(input_path)} sudah sepenuhnya diproses ({skipped_count}/{total_samples} sampel).")
        return 0
        
    if skipped_count > 0:
        print(f"🔄 Menemukan {skipped_count} sampel yang sudah diaugmentasi sebelumnya. Melanjutkan dari sisa data...")
        write_mode = 'a'
    else:
        write_mode = 'w'
        
    if limit is not None:
        # Batasi data yang akan diproses
        remaining_data = remaining_data[:limit]
        print(f"🎯 Membatasi pemrosesan hanya {len(remaining_data)} sampel baru.")
        
    success_count = 0
    
    with open(output_path, write_mode, encoding='utf-8') as out:
        for line_num, entry in tqdm(remaining_data, desc="Mengaugmentasi", unit="sampel"):
            parsed = parse_training_entry(entry)
            if not parsed:
                print(f"\n⚠️ Format data tidak sesuai pada baris ke-{line_num}. Dilewati.")
                continue
                
            try:
                # Memanggil API OpenRouter dengan retry logic
                response = call_llm_with_retry(
                    chain,
                    {
                        "context_docs": parsed["context"],
                        "original_question": parsed["original_question"],
                        "original_thought": parsed["original_thought"],
                        "original_answer": parsed["original_answer"],
                        "format_instructions": parser.get_format_instructions()
                    }
                )
                
                # Cek filter kualitas
                if not quality_filter(response):
                    print(f"\n⚠️ Hasil augmentasi baris ke-{line_num} tidak lolos filter kualitas. Dilewati.")
                    continue
                    
                # Format ulang jawaban dengan tag <thought>
                formatted_thought = f"<thought>\n{response['thought_process']}\n</thought>"
                final_answer = f"{formatted_thought}\n\n{response['answer']}"
                
                # Buat format ShareGPT baru
                augmented_entry = {
                    "conversations": [
                        {
                            "from": "system",
                            "value": "Anda adalah HANIF, asisten AI Ekonomi Syariah. Jawablah pertanyaan pengguna dengan akurat berdasarkan konteks yang diberikan. Mulailah dengan analisis mendalam menggunakan tag <thought>."
                        },
                        {
                            "from": "human",
                            "value": f"Konteks:\n{parsed['context']}\n\nPertanyaan: {response['question']}"
                        },
                        {
                            "from": "gpt",
                            "value": final_answer
                        }
                    ]
                }
                
                # Tulis langsung ke file
                out.write(json.dumps(augmented_entry, ensure_ascii=False) + "\n")
                out.flush()
                success_count += 1
                
                # Beri jeda kecil agar tidak melanggar rate limit OpenRouter
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n❌ Gagal mengaugmentasi baris ke-{line_num}: {e}")
                print("Melanjutkan ke baris berikutnya...")
                time.sleep(2)
                continue
                
    print(f"\n🎉 Selesai memproses file: {os.path.basename(input_path)}")
    print(f"   - Total diproses di sesi ini: {success_count} sampel.")
    print(f"   - Lokasi output: {output_path}")
    return success_count

# =============================
# Main Menu
# =============================
def main():
    parser_arg = argparse.ArgumentParser(description="HANIF Dataset Augmenter (via OpenRouter DeepSeek)")
    parser_arg.add_argument("--source", type=str, choices=["1", "2", "3"], help="Sumber Dataset (1: Semua single, 2: Master file, 3: Spesifik file)")
    parser_arg.add_argument("--model", type=str, choices=["1", "2", "3", "4", "5"], help="Pilihan Model DeepSeek (1-5)")
    parser_arg.add_argument("--custom-model", type=str, help="OpenRouter Model ID kustom (jika opsi model=5)")
    parser_arg.add_argument("--limit", type=int, help="Batasi jumlah sampel yang diaugmentasi per file")
    parser_arg.add_argument("--yes", action="store_true", help="Lewati semua konfirmasi interaktif")
    args = parser_arg.parse_args()

    print("==================================================")
    print("🌟 HANIF DATASET AUGMENTER (via OPENROUTER DEEPSEEK) 🌟")
    print("==================================================")
    
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY tidak ditemukan di environment atau file .env!")
        if args.yes:
            print("❌ Mode non-interaktif aktif, tidak bisa meminta input manual. Program dihentikan.")
            return
        api_key = input("👉 Masukkan OpenRouter API Key Anda: ").strip()
        if not api_key:
            print("❌ API Key tidak valid. Program dihentikan.")
            return
        os.environ["OPENROUTER_API_KEY"] = api_key
    
    # 1. Pilihan Dataset Asal
    source_choice = args.source
    if not source_choice:
        print("\nPilih Sumber Dataset:")
        print("1. Semua file dalam folder 'data/dataset_training_ver2/' (Single per file)")
        print("2. File Master tunggal 'data/MASTER_RAFT_DATASET.jsonl'")
        print("3. Pilih satu file spesifik dari 'data/dataset_training_ver2/'")
        source_choice = input("👉 Masukkan pilihan Anda (1-3): ").strip()
    
    input_files = [] # List of tuples: (input_path, output_path)
    
    if source_choice == "1":
        if not os.path.exists(INPUT_SINGLE_DIR):
            print(f"❌ Folder tidak ditemukan: {INPUT_SINGLE_DIR}")
            return
        
        all_files = [f for f in os.listdir(INPUT_SINGLE_DIR) if f.endswith(".jsonl") and "MASTER" not in f and "augmented" not in f]
        if not all_files:
            print(f"❌ Tidak ada file .jsonl yang cocok di {INPUT_SINGLE_DIR}")
            return
            
        for f in all_files:
            in_path = os.path.join(INPUT_SINGLE_DIR, f)
            clean_name = f.replace(".jsonl", "")
            out_path = os.path.join(OUTPUT_SINGLE_DIR, f"{clean_name}_augmented.jsonl")
            input_files.append((in_path, out_path))
            
        print(f"📂 Ditemukan {len(input_files)} file untuk diproses.")
        
    elif source_choice == "2":
        if not os.path.exists(MASTER_FILE_PATH):
            print(f"❌ File master tidak ditemukan: {MASTER_FILE_PATH}")
            return
        input_files.append((MASTER_FILE_PATH, MASTER_OUTPUT_PATH))
        print("📂 Memilih file master tunggal.")
        
    elif source_choice == "3":
        if not os.path.exists(INPUT_SINGLE_DIR):
            print(f"❌ Folder tidak ditemukan: {INPUT_SINGLE_DIR}")
            return
            
        all_files = [f for f in os.listdir(INPUT_SINGLE_DIR) if f.endswith(".jsonl") and "MASTER" not in f and "augmented" not in f]
        if not all_files:
            print(f"❌ Tidak ada file .jsonl yang cocok di {INPUT_SINGLE_DIR}")
            return
            
        if args.yes:
            print("❌ Opsi 3 (pilih satu file) tidak didukung dalam mode non-interaktif tanpa memilih langsung.")
            return
            
        print("\nDaftar File:")
        for idx, f in enumerate(all_files):
            print(f"{idx + 1}. {f}")
            
        file_idx = input(f"👉 Pilih nomor file (1-{len(all_files)}): ").strip()
        try:
            selected_file = all_files[int(file_idx) - 1]
            in_path = os.path.join(INPUT_SINGLE_DIR, selected_file)
            clean_name = selected_file.replace(".jsonl", "")
            out_path = os.path.join(OUTPUT_SINGLE_DIR, f"{clean_name}_augmented.jsonl")
            input_files.append((in_path, out_path))
            print(f"📂 Memilih file: {selected_file}")
        except Exception:
            print("❌ Pilihan tidak valid.")
            return
    else:
        print("❌ Pilihan tidak valid.")
        return
        
    # 2. Pilihan Model DeepSeek
    model_choice = args.model
    if not model_choice:
        print("\nPilih Model DeepSeek:")
        for key, (val, desc) in MODEL_OPTIONS.items():
            print(f"{key}. {desc} [{val}]")
        print("5. Input Model ID kustom secara manual")
        model_choice = input("👉 Pilih model (1-5) [Default 1]: ").strip()
        if not model_choice:
            model_choice = "1"
        
    if model_choice in MODEL_OPTIONS:
        selected_model = MODEL_OPTIONS[model_choice][0]
    elif model_choice == "5":
        if args.yes and args.custom_model:
            selected_model = args.custom_model
        else:
            selected_model = input("👉 Masukkan OpenRouter Model ID (misal: deepseek/deepseek-chat): ").strip()
            if not selected_model:
                selected_model = "deepseek/deepseek-chat"
    else:
        selected_model = MODEL_OPTIONS["1"][0]
        
    print(f"🤖 Menggunakan model: {selected_model}")
    
    # 3. Batasan Jumlah Soal (Limit)
    limit = args.limit
    if limit is None:
        if args.yes:
            limit = None
        else:
            limit_input = input("\n👉 Batasi jumlah sampel yang diaugmentasi per file? (Ketik angka, atau kosongkan untuk SEMUA): ").strip()
            if limit_input:
                try:
                    limit = int(limit_input)
                    print(f"🎯 Batasan aktif: Maksimal {limit} sampel per file.")
                except ValueError:
                    print("⚠️ Input tidak valid. Memproses seluruh data.")
    else:
        print(f"🎯 Batasan aktif: Maksimal {limit} sampel per file.")
            
    # Konfirmasi sebelum jalan
    if not args.yes:
        confirm = input("\n👉 Siap memulai augmentasi? (y/n) [y]: ").strip().lower()
        if confirm not in ['', 'y', 'yes']:
            print("❌ Dibatalkan oleh pengguna.")
            return
        
    # Inisialisasi LLM & Chain
    print("\n⏳ Menginisialisasi OpenRouter LLM...")
    try:
        llm = get_llm(selected_model)
    except Exception as e:
        print(f"❌ Gagal membuat objek LLM: {e}")
        return
    
    # Bangun Prompt & Parser
    raft_prompt = ChatPromptTemplate.from_messages([
        ("system", raft_augment_system_prompt),
        ("user", "Silakan augmentasikan QA di atas dengan standar tinggi HANIF.")
    ])
    
    chain = raft_prompt | llm
    
    total_success = 0
    
    # Jalankan loop pemrosesan
    for in_path, out_path in input_files:
        print(f"\n=========================================")
        print(f"🚀 Memulai pemrosesan untuk: {os.path.basename(in_path)}")
        print(f"=========================================")
        
        success = process_file(in_path, out_path, chain, limit=limit)
        total_success += success
        
    print("\n==================================================")
    print("✨ PROSES AUGMENTASI SELESAI! ✨")
    print(f"👉 Total sampel berhasil diaugmentasi: {total_success} sampel.")
    print("==================================================")

if __name__ == "__main__":
    main()
