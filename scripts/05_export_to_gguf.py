import os
# Fix for missing CUDA 13 libraries on some systems
import ctypes
# Get absolute path to venv relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.abspath(os.path.join(script_dir, "../venv"))
cuda_lib_path = os.path.join(venv_path, "lib/python3.14/site-packages/nvidia/cu13/lib/libnvJitLink.so.13")

if os.path.exists(cuda_lib_path):
    try:
        # Load the library globally so other libraries can see it
        ctypes.CDLL(cuda_lib_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Warning: Could not pre-load CUDA library: {e}")

# Monkey-patch Hasher to avoid dill/pickle error on Python 3.14
try:
    from datasets.fingerprint import Hasher
    Hasher.hash = lambda *args, **kwargs: "dummy_hash"
except ImportError:
    pass

from unsloth import FastLanguageModel
import torch
...
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