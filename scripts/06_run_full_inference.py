import json
import os
import argparse
from langchain_ollama import ChatOllama
from tqdm import tqdm
from datetime import datetime

# --- CONFIG ---
# Mapping model names to their training metadata for naming
MODEL_METADATA = {
    "hanif_strat1_retrain:latest": "strat1_lr5e5_ep2_20260602_1659",
    "hanif_strat2_retrain:latest": "strat2_lr5e5_ep2_20260602_1802",
    "hanif_strat3:latest": "strat3_lr2e4_ep1_original" # Strat 3 was from previous session
}

STRATEGIES = {
    "strat1": {
        "dataset": "data/dataset_splits/strat1_pure_aug_test.jsonl",
        "model": "hanif_strat1_retrain:latest"
    },
    "strat2": {
        "dataset": "data/dataset_splits/strat2_cross_test.jsonl",
        "model": "hanif_strat2_retrain:latest"
    },
    "strat3": {
        "dataset": "data/dataset_splits/strat3_dual_test.jsonl",
        "model": "hanif_strat3:latest"
    }
}
OUTPUT_DIR = "eval_results_full"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_inference(strategy_id):
    config = STRATEGIES.get(strategy_id)
    if not config:
        print(f"⚠️ Strategy ID {strategy_id} not found in config.")
        return

    dataset_path = config["dataset"]
    model_name = config["model"]
    metadata_tag = MODEL_METADATA.get(model_name, strategy_id)

    if not os.path.exists(dataset_path):
        print(f"⚠️ Dataset for {strategy_id} not found at {dataset_path}")
        return

    print(f"\n" + "="*50)
    print(f"🚀 Memulai Full Inference: {strategy_id.upper()}")
    print(f"🤖 Model: {model_name}")
    print(f"🏷️ Metadata: {metadata_tag}")
    print(f"📂 Dataset: {dataset_path}")
    print("="*50 + "\n")
    
    llm = ChatOllama(model=model_name, temperature=0.0)

    output_filename = f"results_{metadata_tag}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # --- RESUME LOGIC ---
    results = []
    processed_indices = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f_existing:
                results = json.load(f_existing)
                processed_indices = {item["query_index"] for item in results}
                print(f"⏩ Resuming: {len(processed_indices)} queries already processed.")
        except Exception as e:
            print(f"⚠️ Could not read existing results, starting fresh: {e}")
            results = []

    with open(dataset_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc=f"Processing {strategy_id}")):
            query_idx = i + 1
            if query_idx in processed_indices:
                continue # Skip already processed
                
            data = json.loads(line)
            human_msg = data["conversations"][1]["value"]
            ground_truth = data["conversations"][2]["value"]
            
            try:
                response = llm.invoke(human_msg)
                model_answer = response.content
            except Exception as e:
                print(f"\n❌ Error invoking LLM for {strategy_id} at query {query_idx}: {e}")
                model_answer = "ERROR"

            results.append({
                "query_index": query_idx,
                "question_with_context": human_msg,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "model_used": model_name,
                "metadata": metadata_tag
            })
            
            # Incremental save
            with open(output_path, "w") as f_out:
                json.dump(results, f_out, indent=4)
    
    print(f"\n✅ Selesai {strategy_id}! Results: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Full batch inference for Ekonomi Syariah Chatbot.")
    parser.add_argument(
        "--strat", 
        nargs="+", 
        choices=list(STRATEGIES.keys()), 
        default=list(STRATEGIES.keys()),
        help="List strategy IDs to run"
    )
    
    args = parser.parse_args()
    
    for strategy_id in args.strat:
        run_inference(strategy_id)

if __name__ == "__main__":
    main()
