import json
import os
import re

DATASET_DIR = "data/dataset_training_ver2/"

# Daftar frasa yang ingin dihapus (case-insensitive)
NOISE_PATTERNS = [
    r"^Menurut teks(?: tersebut)?,?\s*",
    r"^Berdasarkan (?:teks|dokumen|referensi|konteks)(?: tersebut| yang diberikan)?,?\s*",
    r"^Sesuai dengan (?:teks|dokumen|referensi|konteks),?\s*",
    r"^Teks menyebutkan bahwa\s*",
    r"^Dalam dokumen ini,?\s*",
    r"^Disebutkan bahwa\s*",
    r"^Berdasarkan data yang tersedia,?\s*",
]

IMAGE_PATTERNS = [
    r"\[IMAGE\]",
    r"\[Gambar\]",
    r"!\[.*?\]\(.*?\)",  # Markdown images
    r"\(lihat gambar.*?\)",
    r"\(tabel \d+.*?\)", # Seringkali tabel juga rusak di OCR
]

def clean_noise(text):
    # 1. Hapus referensi gambar
    for pattern in IMAGE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 2. Hapus link/URL
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # 3. Hapus simbol titik-titik yang banyak (2 atau lebih)
    text = re.sub(r'\.{2,}', ' ', text)
    
    # 4. Normalisasi separator dan simbol berlebihan
    text = re.sub(r'-{4,}', '---', text)
    text = re.sub(r'={4,}', '===', text)
    text = re.sub(r'_{4,}', '___', text)
    text = re.sub(r'\*{4,}', '***', text)
    
    # 5. Perbaiki spasi pada kata berhubung (misal: "unsur - unsur" -> "unsur-unsur")
    text = re.sub(r'(\w+)\s+-\s+(\w+)', r'\1-\2', text)

    # 6. Hapus kata atau frase berulang (misal: "halal halal" -> "halal")
    # Ulangi 2x untuk menangkap pengulangan frase yang kompleks (misal: "A B A B" -> "A B")
    for _ in range(2):
        text = re.sub(r'(?<!\S)(\S+(?:\s+\S+){0,5})(?:\s+\1(?!\S))+', r'\1', text, flags=re.IGNORECASE)
    
    # 7. Hapus spasi berlebih, tab, dan newline berulang
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def clean_gpt_response(text):
    # Pisahkan thought dan answer
    parts = text.split("</thought>")
    if len(parts) < 2:
        return clean_noise(text)
    
    thought = clean_noise(parts[0]) + "</thought>"
    answer = clean_noise(parts[1])
    
    # Bersihkan noise pembuka di bagian answer
    for pattern in NOISE_PATTERNS:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)
    
    # Capitalize huruf pertama
    if answer:
        answer = answer[0].upper() + answer[1:]
        
    return f"{thought}\n\n{answer}"

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Folder {DATASET_DIR} tidak ditemukan.")
        return

    files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jsonl")]
    
    for filename in files:
        path = os.path.join(DATASET_DIR, filename)
        new_data = []
        
        print(f"🧹 Membersihkan: {filename}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                
                # 1. Bersihkan Konteks (Human)
                item["conversations"][1]["value"] = clean_noise(item["conversations"][1]["value"])
                
                # 2. Bersihkan Respon (GPT)
                item["conversations"][2]["value"] = clean_gpt_response(item["conversations"][2]["value"])
                
                new_data.append(item)
        
        # Simpan kembali
        with open(path, "w", encoding="utf-8") as f:
            for item in new_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("✨ Dataset bersih dari noise gambar, spasi, dan meta-references.")

if __name__ == "__main__":
    main()
