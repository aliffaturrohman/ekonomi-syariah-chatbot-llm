import os
import json
import torch
import ctypes
# Fix for missing CUDA 13 libraries on some systems
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.abspath(os.path.join(script_dir, "../venv"))
cuda_lib_path = os.path.join(venv_path, "lib/python3.14/site-packages/nvidia/cu13/lib/libnvJitLink.so.13")
if os.path.exists(cuda_lib_path):
    try:
        ctypes.CDLL(cuda_lib_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Warning: Could not pre-load CUDA library: {e}")

from unsloth import FastLanguageModel
from tqdm import tqdm

# --- CONFIG ---
ADAPTER_PATH = "ekonomi-syariah-chatbot-llm/models/adapters/qwen_raft_ekonomi_syariah_strat1_final_r32_lr2e5_ep3_20260604_1730"
DATASET_PATH = "ekonomi-syariah-chatbot-llm/data/dataset_splits/strat1_pure_aug_test.jsonl"
OUTPUT_DIR = "ekonomi-syariah-chatbot-llm/eval_results_full"
MAX_SEQ_LENGTH = 2048
METADATA_TAG = "strat1_final_r32_lr2e5_ep3_20260604_1730"

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
    FastLanguageModel.for_inference(model)

    # 2. Setup Output Path
    output_path = os.path.join(OUTPUT_DIR, f"results_{METADATA_TAG}.json")
    
    # 3. Load Dataset
    with open(DATASET_PATH, "r") as f:
        lines = f.readlines()

    # Limit to 5 samples
    print(f"📊 Processing 5 samples from: {DATASET_PATH}")
    sample_lines = lines[:5]

    results = []

    # 4. Process
    for i, line in enumerate(tqdm(sample_lines, desc="Inference Strat1")):
        query_idx = i + 1
        data = json.loads(line)
        human_msg = data["conversations"][1]["value"]
        ground_truth = data["conversations"][2]["value"]

        # Apply chat template
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
                temperature = 0.1,
                top_p = 0.9,
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

        results.append({
            "query_index": query_idx,
            "question_with_context": human_msg,
            "ground_truth": ground_truth,
            "model_answer": generated_text,
            "strategy": "strat1_eval_test"
        })

    # Save final results
    with open(output_path, "w") as f_out:
        json.dump(results, f_out, indent=4)

    print(f"✅ Selesai! Hasil disimpan di: {output_path}")

if __name__ == "__main__":
    run_direct_inference()
