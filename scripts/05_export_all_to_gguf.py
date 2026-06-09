import os
import ctypes
from unsloth import FastLanguageModel
import torch

# Fix for missing CUDA 13 libraries
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.abspath(os.path.join(script_dir, "../venv"))
cuda_lib_path = os.path.join(venv_path, "lib/python3.14/site-packages/nvidia/cu13/lib/libnvJitLink.so.13")

if os.path.exists(cuda_lib_path):
    try:
        ctypes.CDLL(cuda_lib_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Warning: Could not pre-load CUDA library: {e}")

# Monkey-patch Hasher for Python 3.14 compatibility
try:
    from datasets.fingerprint import Hasher
    Hasher.hash = lambda *args, **kwargs: "dummy_hash"
except ImportError:
    pass

# Monkey-patch Unsloth's internet/sudo check to bypass the failure
import unsloth_zoo.llama_cpp
unsloth_zoo.llama_cpp.do_we_need_sudo = lambda *args, **kwargs: False
unsloth_zoo.llama_cpp.IS_COLAB_ENVIRONMENT = True
unsloth_zoo.llama_cpp.IS_KAGGLE_ENVIRONMENT = True

# --- CONFIG ---
STRATEGIES = ["strat1", "strat2", "strat3"]
QUANTIZATION_METHOD = "q4_k_m"

def export_model(strategy_name):
    adapter_dir = os.path.abspath(os.path.join(script_dir, f"../models/adapters/qwen_raft_ekonomi_syariah_{strategy_name}"))
    
    if not os.path.exists(adapter_dir):
        print(f"⚠️ Adapter directory not found: {adapter_dir}")
        return

    print(f"\n🔄 Loading LoRA Adapter from: {adapter_dir}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_dir,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    print(f"💾 Converting to GGUF format ({QUANTIZATION_METHOD}) for {strategy_name}...")
    
    # This will save the GGUF file inside the adapter_dir
    model.save_pretrained_gguf(
        adapter_dir,
        tokenizer,
        quantization_method = QUANTIZATION_METHOD
    )
    
    print(f"✅ Finished exporting {strategy_name}!")

def main():
    for strat in STRATEGIES:
        export_model(strat)

if __name__ == "__main__":
    main()
