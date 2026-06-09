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

# --- CONFIG NEW MODELS ---
NEW_ADAPTERS = [
    "models/adapters/qwen_raft_ekonomi_syariah_strat1_lr5e5_ep2_20260602_1659",
    "models/adapters/qwen_raft_ekonomi_syariah_strat2_lr5e5_ep2_20260602_1802"
]
QUANTIZATION_METHOD = "q4_k_m"

def export_model(adapter_path):
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    full_adapter_path = os.path.join(project_root, adapter_path)
    
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

    # We'll save the GGUF in a sibling directory ending in _gguf
    output_gguf_dir = full_adapter_path + "_gguf"
    
    print(f"💾 Converting to GGUF format ({QUANTIZATION_METHOD})...")
    
    model.save_pretrained_gguf(
        output_gguf_dir,
        tokenizer,
        quantization_method = QUANTIZATION_METHOD
    )
    
    print(f"✅ Finished exporting to: {output_gguf_dir}")

def main():
    for adapter in NEW_ADAPTERS:
        export_model(adapter)

if __name__ == "__main__":
    main()
