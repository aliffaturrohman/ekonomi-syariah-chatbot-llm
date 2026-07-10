import os
import json
import torch
import ctypes
from unsloth import FastLanguageModel
from tqdm import tqdm
import time

# --- CONFIG ---
ADAPTER_PATH = "ekonomi-syariah-chatbot-llm/models/adapters/qwen_raft_ekonomi_syariah_strat1_final_r32_lr2e5_ep3_20260604_1730"
DATASET_PATH = "ekonomi-syariah-chatbot-llm/data/dataset_splits/strat1_pure_aug_test.jsonl"
MAX_SEQ_LENGTH = 2048

def run_stress_test():
    # Fix for missing CUDA 13 libraries
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_path = os.path.abspath(os.path.join(script_dir, "../venv"))
    cuda_lib_path = os.path.join(venv_path, "lib/python3.14/site-packages/nvidia/cu13/lib/libnvJitLink.so.13")
    if os.path.exists(cuda_lib_path):
        try:
            ctypes.CDLL(cuda_lib_path, mode=ctypes.RTLD_GLOBAL)
        except:
            pass

    print(f"🚀 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = ADAPTER_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    with open(DATASET_PATH, "r") as f:
        lines = f.readlines()[:5]

    print(f"🔥 Starting Inference Loop (5 samples)...")
    for i, line in enumerate(lines):
        data = json.loads(line)
        human_msg = data["conversations"][1]["value"]

        messages = [
            {"role": "system", "content": "Anda adalah HANIF. Jawablah dengan akurat."},
            {"role": "user", "content": human_msg}
        ]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids = inputs,
                max_new_tokens = 512, # Cukup panjang untuk mengisi KV Cache
                use_cache = True,
                temperature = 0.1,
            )
        print(f"✅ Processed sample {i+1}")
        
    print("🏁 Stress test finished for this instance.")

if __name__ == "__main__":
    run_stress_test()
