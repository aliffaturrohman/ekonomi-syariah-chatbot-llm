import json
import random
import os
import glob

INPUT_DIR = "eval_results_full"
OUTPUT_DIR = "eval_results_samples"
SAMPLE_COUNT = 100

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_samples():
    # Mencari semua file hasil inference yang ada
    result_files = glob.glob(f"{INPUT_DIR}/results_*.json")
    
    if not result_files:
        print("⚠️  Tidak ada file hasil inference ditemukan di eval_results_full/")
        return

    for file_path in result_files:
        file_name = os.path.basename(file_path)
        print(f"🎲 Mengambil {SAMPLE_COUNT} sampel acak dari: {file_name}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        if len(data) <= SAMPLE_COUNT:
            sampled_data = data
            print(f"ℹ️  Data asli hanya {len(data)}, mengambil semuanya.")
        else:
            # Gunakan seed agar hasil sampling bisa direproduksi (penting untuk TA)
            random.seed(42) 
            sampled_data = random.sample(data, SAMPLE_COUNT)
        
        output_path = os.path.join(OUTPUT_DIR, file_name)
        with open(output_path, "w") as f_out:
            json.dump(sampled_data, f_out, indent=4)
        
        print(f"✅ Sampel disimpan ke: {output_path}")

if __name__ == "__main__":
    create_samples()
