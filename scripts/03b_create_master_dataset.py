import os

# --- KONFIGURASI ---
INPUT_DIR = "../data/dataset_training_ver2/"
OUTPUT_FILE = "../data/dataset_training_ver2/MASTER_RAFT_DATASET.jsonl"

def main():
    print("🔄 Sedang menggabungkan semua file dataset...")
    
    # Cek folder input
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Folder input tidak ditemukan: {INPUT_DIR}")
        return

    all_lines = []
    file_count = 0
    
    # 1. Loop semua file di folder dataset
    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)
        
        # Filter: Hanya file .jsonl dan JANGAN baca file output (MASTER) itu sendiri
        if filename.endswith(".jsonl") and "MASTER_RAFT_DATASET" not in filename:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    # Cek apakah file kosong
                    if not lines:
                        print(f"   ⚠️  File kosong (dilewati): {filename}")
                        continue
                        
                    all_lines.extend(lines)
                    file_count += 1
                    print(f"   + Mengambil {len(lines)} baris dari: {filename}")
            except Exception as e:
                print(f"   ❌ Gagal membaca {filename}: {e}")

    # 2. Cek apakah ada data yang terkumpul
    if not all_lines:
        print("\n❌ Tidak ada data yang ditemukan untuk digabung!")
        return

    # 3. Simpan ke file MASTER
    print(f"\n💾 Menyimpan ke {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)
        
        print(f"✅ SUKSES! Total dataset: {len(all_lines)} sampel data.")
        print("   File ini siap digunakan untuk script '04_train_model.py'")
        
    except Exception as e:
        print(f"❌ Gagal menyimpan file master: {e}")

if __name__ == "__main__":
    main()