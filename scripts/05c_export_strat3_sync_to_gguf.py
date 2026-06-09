import os
import ctypes
from unsloth import FastLanguageModel
import torch

# Fix for missing CUDA 13 libraries
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.abspath(os.path.join(script_dir, "../venv"))
cuda_lib_path = "/home/alif-faturrohman/coding/ekonomi-syariah-chatbot-llm/venv/lib/python3.14/site-packages/nvidia/cu13/lib/libnvJitLink.so.13"

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

# Monkey-patch Unsloth interactive checks
import unsloth_zoo.llama_cpp
unsloth_zoo.llama_cpp.do_we_need_sudo = lambda *args, **kwargs: False
unsloth_zoo.llama_cpp.IS_COLAB_ENVIRONMENT = True
unsloth_zoo.llama_cpp.IS_KAGGLE_ENVIRONMENT = True

# --- CONFIG ---
ADAPTER_PATH = "models/adapters/qwen_raft_ekonomi_syariah_strat3_lr5e5_ep2_20260604_0943"
QUANTIZATION_METHOD = "q4_k_m"

def export_model():
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    full_adapter_path = os.path.join(project_root, ADAPTER_PATH)
    
    if not os.path.exists(full_adapter_path):
        print(f"⚠️ Adapter directory not found: {full_adapter_path}")
        return

    print(f"\n" + "="*50)
    print(f"🔄 Loading LoRA Adapter from: {full_adapter_path}")
    print("="*50)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = full_adapter_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    output_gguf_dir = full_adapter_path + "_gguf"
    print(f"💾 Converting to GGUF format ({QUANTIZATION_METHOD})...")
    
    model.save_pretrained_gguf(
        output_gguf_dir,
        tokenizer,
        quantization_method = QUANTIZATION_METHOD
    )
    
    print(f"✅ Finished! GGUF saved to: {output_gguf_dir}")

if __name__ == "__main__":
    export_model()
