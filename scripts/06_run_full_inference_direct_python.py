import os
import json
import torch
import ctypes

# Fix for missing CUDA 13 libraries on some systems
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.abspath(os.path.join(script_dir, "../venv"))
cuda_lib_path = "/home/alif-faturrohman/coding/ekonomi-syariah-chatbot-llm/venv/lib/python3.14/site-packages/nvidia/cu13/lib/libnvJitLink.so.13"
if os.path.exists(cuda_lib_path):
    try:
        ctypes.CDLL(cuda_lib_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Warning: Could not pre-load CUDA library: {e}")

from unsloth import FastLanguageModel
from tqdm import tqdm
from datetime import datetime

# --- CONFIG ---
ADAPTER_PATH = "models/adapters/qwen_raft_ekonomi_syariah_strat3_lr5e5_ep2_20260604_0943"
DATASET_PATH = "data/dataset_splits/strat3_dual_test.jsonl"
OUTPUT_DIR = "eval_results_full"
MAX_SEQ_LENGTH = 2048

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_direct_inference():
    print(f"🚀 Loading model directly from adapter: {ADAPTER_PATH}")
    
    # 1. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = ADAPTER_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    # 2. Setup Output Path (Using metadata from ADAPTER_PATH for consistency)
    # Tag: strat3_lr5e5_ep2_20260604_0943
    metadata_tag = "strat3_lr5e5_ep2_20260604_0943"
    output_path = os.path.join(OUTPUT_DIR, f"results_{metadata_tag}.json")
    
    # 3. Resume Logic
    results = []
    processed_indices = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
            processed_indices = {item["query_index"] for item in results}
            print(f"⏩ Resuming: {len(processed_indices)} queries already processed.")

    # 4. Load Dataset
    with open(DATASET_PATH, "r") as f:
        lines = f.readlines()

    print(f"📊 Total queries to process: {len(lines)}")

    # 5. Process
    for i, line in enumerate(tqdm(lines, desc="Inference Strat3")):
        query_idx = i + 1
        if query_idx in processed_indices:
            continue

        data = json.loads(line)
        human_msg = data["conversations"][1]["value"]
        ground_truth = data["conversations"][2]["value"]

        # Apply chat template manually
        messages = [
            {"role": "system", "content": "Anda adalah HANIF, asisten AI Ekonomi Syariah. Jawablah pertanyaan pengguna dengan akurat berdasarkan konteks yang diberikan. Mulailah dengan analisis mendalam menggunakan tag <thought>."},
            {"role": "user", "content": human_msg}
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids = inputs,
                max_new_tokens = 1500,
                use_cache = True,
                temperature = 0.1, # Low temp for deterministic evaluation
                top_p = 0.9,
            )
        
        # Decode only the NEW tokens
        generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

        results.append({
            "query_index": query_idx,
            "question_with_context": human_msg,
            "ground_truth": ground_truth,
            "model_answer": generated_text,
            "strategy": "strat3_sync_direct"
        })

        # Save incrementally
        with open(output_path, "w") as f_out:
            json.dump(results, f_out, indent=4)

    print(f"✅ Selesai! Hasil disimpan di: {output_path}")
    return output_path

if __name__ == "__main__":
    run_direct_inference()
