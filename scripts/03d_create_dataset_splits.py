import os
import json

# --- KONFIGURASI PATH DINAMIS ---
# Memastikan script bisa dipanggil dari folder mana pun dengan aman
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
SPLIT_DIR = os.path.join(DATA_DIR, "dataset_splits")
MASTER_PATH = os.path.join(DATA_DIR, "MASTER_RAFT_DATASET.jsonl")
AUGMENTED_PATH = os.path.join(DATA_DIR, "MASTER_RAFT_DATASET_AUGMENTED.jsonl")

def read_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"❌ File tidak ditemukan: {filepath}")
        return data
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(data, filepath):
    # Pastikan direktori tujuan ada
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Berhasil menulis {len(data)} sampel ke: data/dataset_splits/{os.path.basename(filepath)}")

def main():
    print("==================================================")
    print("🌟 HANIF DATASET SPLITTER (3 STRATEGI TRAINING) 🌟")
    print("==================================================")
    
    print("\n📖 Membaca master datasets...")
    master_data = read_jsonl(MASTER_PATH)
    augmented_data = read_jsonl(AUGMENTED_PATH)
    
    total_master = len(master_data)
    total_aug = len(augmented_data)
    
    print(f"  Total Master samples: {total_master}")
    print(f"  Total Augmented samples: {total_aug}")
    
    if total_master == 0 or total_aug == 0:
        print("❌ Error: Salah satu atau kedua dataset kosong atau tidak ditemukan!")
        return
        
    if total_master != total_aug:
        print("❌ Error: Jumlah baris antara Master dan Augmented tidak selaras!")
        return

    # Tentukan batas split 80% secara konsisten
    # 80% dari 1454 = 1163 sampel
    split_idx = 1163 
    
    # ----------------------------------------------------
    # STRATEGI 1: Pure Augmented
    # ----------------------------------------------------
    print("\n--- Membuat Dataset Strategi 1 (Pure Augmented) ---")
    s1_train = augmented_data[:split_idx]
    s1_test = augmented_data[split_idx:]
    
    s1_train_path = os.path.join(SPLIT_DIR, "strat1_pure_aug_train.jsonl")
    s1_test_path = os.path.join(SPLIT_DIR, "strat1_pure_aug_test.jsonl")
    
    write_jsonl(s1_train, s1_train_path)
    write_jsonl(s1_test, s1_test_path)
    
    # ----------------------------------------------------
    # STRATEGI 2: Cross-Distribution (Train Augmented, Test Raw)
    # ----------------------------------------------------
    print("\n--- Membuat Dataset Strategi 2 (Cross-Distribution) ---")
    s2_train = augmented_data[:split_idx]
    s2_test = master_data[split_idx:]
    
    s2_train_path = os.path.join(SPLIT_DIR, "strat2_cross_train.jsonl")
    s2_test_path = os.path.join(SPLIT_DIR, "strat2_cross_test.jsonl")
    
    write_jsonl(s2_train, s2_train_path)
    write_jsonl(s2_test, s2_test_path)
    
    # ----------------------------------------------------
    # STRATEGI 3: Dual-Prompt (Train Raw + Augmented, Test Augmented)
    # ----------------------------------------------------
    print("\n--- Membuat Dataset Strategi 3 (Dual-Prompt) ---")
    s3_train_raw = master_data[:split_idx]
    s3_train_aug = augmented_data[:split_idx]
    
    # Gabungkan secara berselang-seling agar distribusinya merata saat training
    s3_train = []
    for r, a in zip(s3_train_raw, s3_train_aug):
        s3_train.append(r)
        s3_train.append(a)
        
    s3_test = augmented_data[split_idx:]
    
    s3_train_path = os.path.join(SPLIT_DIR, "strat3_dual_train.jsonl")
    s3_test_path = os.path.join(SPLIT_DIR, "strat3_dual_test.jsonl")
    
    write_jsonl(s3_train, s3_train_path)
    write_jsonl(s3_test, s3_test_path)
    
    print("\n==================================================")
    print("✨ PEMBAGIAN DATASET SELESAI! ✨")
    print(f"👉 Lokasi Folder Output: {SPLIT_DIR}")
    print("==================================================")

if __name__ == "__main__":
    main()
