import os
import json
import ctypes
import gc
import sys
from tqdm import tqdm

def setup_env():
    # 1. Identify all NVIDIA library paths in venv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_site = os.path.abspath(os.path.join(script_dir, "../venv/lib/python3.14/site-packages"))
    nvidia_base = os.path.join(venv_site, "nvidia")
    
    if not os.path.exists(nvidia_base):
        return

    lib_paths = []
    for root, dirs, files in os.walk(nvidia_base):
        if 'lib' in dirs:
            lib_paths.append(os.path.join(root, 'lib'))
    
    if not lib_paths:
        return
        
    new_ld_path = ":".join(lib_paths)
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    
    # Check if we already have these paths
    if all(p in current_ld_path for p in lib_paths):
        return

    # 2. Update environment and RE-EXECUTE
    print(f"🔧 Setting LD_LIBRARY_PATH and re-executing script...")
    os.environ["LD_LIBRARY_PATH"] = f"{new_ld_path}:{current_ld_path}"
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"❌ Failed to re-exec: {e}")

# IMPORTANT: Call setup_env BEFORE any other imports that might trigger CUDA
setup_env()

# --- CONFIG ---
INPUT_DIR = "ekonomi-syariah-chatbot-llm/data/dataset_splits"
OUTPUT_DIR = "ekonomi-syariah-chatbot-llm/eval_results_full"
ADAPTER_BASE_DIR = "ekonomi-syariah-chatbot-llm/models/adapters"
MAX_SEQ_LENGTH = 2048

def run_single_inference(job):
    import torch
    from unsloth import FastLanguageModel
    
    strategy = job["strategy"]
    metadata_tag = job["param_tag"]
    
    if metadata_tag.startswith(strategy):
        adapter_path = os.path.join(ADAPTER_BASE_DIR, f"qwen_raft_ekonomi_syariah_{metadata_tag}")
    else:
        adapter_path = os.path.join(ADAPTER_BASE_DIR, f"qwen_raft_ekonomi_syariah_{strategy}_{metadata_tag}")
    
    # Mapping dataset
    dataset_map = {
        "strat1": "strat1_pure_aug_test.jsonl",
        "strat2": "strat2_cross_test.jsonl",
        "strat3": "strat3_dual_test.jsonl"
    }
    dataset_path = os.path.join(INPUT_DIR, dataset_map.get(strategy, "strat1_pure_aug_test.jsonl"))
    output_path = os.path.join(OUTPUT_DIR, f"results_{metadata_tag}.json")

    if not os.path.exists(adapter_path):
        print(f"⚠️ Adapter not found: {adapter_path}")
        return

    print(f"🚀 [Job {metadata_tag}] Loading model from {adapter_path}...")
    
    # Force clear before loading
    torch.cuda.empty_cache()
    gc.collect()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # Resume logic
    results = []
    processed_indices = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            try:
                results = json.load(f)
                processed_indices = {item["query_index"] for item in results}
            except:
                pass

    with open(dataset_path, "r") as f:
        lines = f.readlines()

    print(f"📊 [Job {metadata_tag}] Processing {len(lines) - len(processed_indices)} remaining queries...")

    for i, line in enumerate(tqdm(lines, desc=f"Inference {metadata_tag}")):
        query_idx = i + 1
        if query_idx in processed_indices:
            continue

        data = json.loads(line)
        human_msg = data["conversations"][1]["value"]
        ground_truth = data["conversations"][2]["value"]

        messages = [
            {"role": "system", "content": "Anda adalah HANIF, asisten AI Ekonomi Syariah. Jawablah dengan akurat dan awali dengan analisis mendalam di <thought>."},
            {"role": "user", "content": human_msg}
        ]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids = inputs,
                max_new_tokens = 1024,
                use_cache = True,
                temperature = 0.1,
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

        results.append({
            "query_index": query_idx,
            "question_with_context": human_msg,
            "ground_truth": ground_truth,
            "model_answer": generated_text,
            "strategy": strategy
        })

        # Periodic save
        if query_idx % 5 == 0:
            with open(output_path, "w") as f_out:
                json.dump(results, f_out, indent=4)

    # Final save
    with open(output_path, "w") as f_out:
        json.dump(results, f_out, indent=4)
    
    print(f"✅ [Job {metadata_tag}] Selesai!")
    
    # Cleanup VRAM
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    # Detect jobs from parameter_end_training.json
    param_path = "ekonomi-syariah-chatbot-llm/scripts/parameter_end_training.json"
    with open(param_path, "r") as f:
        jobs = json.load(f)

    EXPECTED_COUNT = 291
    pending_jobs = []
    for job in jobs:
        tag = job.get("param_tag")
        if job.get("status", "").startswith("Sukses") and tag:
            output_file = os.path.join(OUTPUT_DIR, f"results_{tag}.json")
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        current_results = json.load(f)
                        if len(current_results) < EXPECTED_COUNT:
                            pending_jobs.append(job)
                except:
                    pending_jobs.append(job)
            else:
                pending_jobs.append(job)

    if not pending_jobs:
        print("✅ No pending inference jobs found.")
        return

    print(f"🔥 Found {len(pending_jobs)} pending jobs. Running sequentially...")
    
    for job in pending_jobs:
        try:
            run_single_inference(job)
        except Exception as e:
            print(f"❌ Error in job {job['param_tag']}: {e}")
            import torch
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
