from unsloth import FastLanguageModel
import torch
import os

# --- KONFIGURASI ---
# Path tempat hasil training (LoRA) disimpan kemarin
ADAPTER_DIR = "../models/adapters/qwen_raft_ekonomi_syariah_ver2"

# Format Quantization (q4_k_m adalah standar terbaik untuk balance speed/quality)
QUANTIZATION_METHOD = "q4_k_m" 

def main():
    print(f"🔄 Loading LoRA Adapter dari: {ADAPTER_DIR}")
    
    # 1. Load Model & Adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = ADAPTER_DIR,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # 2. Simpan ke GGUF
    # Unsloth akan otomatis download model base, merge lora, dan convert ke GGUF
    print(f"💾 Sedang mengonversi ke format GGUF ({QUANTIZATION_METHOD})...")
    print("   (Ini butuh waktu beberapa menit dan RAM yang cukup)")
    
    model.save_pretrained_gguf(
        ADAPTER_DIR, # File GGUF akan muncul di folder ini juga
        tokenizer,
        quantization_method = QUANTIZATION_METHOD
    )
    
    print(f"✅ Selesai! File GGUF tersimpan di dalam folder: {ADAPTER_DIR}")
    print("   Cari file berakhiran '.gguf' di sana.")

if __name__ == "__main__":
    main()