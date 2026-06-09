import json
import os
import argparse
from langchain_ollama import ChatOllama
from tqdm import tqdm

# --- CONFIG ---
STRATEGIES = {
    "strat1_retrain": {
        "dataset": "data/dataset_splits/strat1_pure_aug_test.jsonl",
        "model": "hanif_strat1_retrain:latest"
    },
    "strat2_retrain": {
        "dataset": "data/dataset_splits/strat2_cross_test.jsonl",
        "model": "hanif_strat2_retrain:latest"
    },
    "strat3": {
        "dataset": "data/dataset_splits/strat3_dual_test.jsonl",
        "model": "hanif_strat3:latest"
    }
}
OUTPUT_DIR = "eval_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_inference(strategy_id):
    config = STRATEGIES.get(strategy_id)
    if not config:
        print(f"⚠️ Strategy ID {strategy_id} not found in config.")
        return

    dataset_path = config["dataset"]
    model_name = config["model"]

    if not os.path.exists(dataset_path):
        print(f"⚠️ Dataset for {strategy_id} not found at {dataset_path}")
        return

    print(f"\n" + "="*50)
    print(f"🚀 Memulai Inference: {strategy_id.upper()}")
    print(f"🤖 Model: {model_name}")
    print(f"📂 Dataset: {dataset_path}")
    print("="*50 + "\n")
    
    llm = ChatOllama(model=model_name, temperature=0.0)

    results = []
    output_path = os.path.join(OUTPUT_DIR, f"{strategy_id}_results.json")
    
    with open(dataset_path, "r") as f:
        lines = f.readlines()
        # Limit to 5 samples as requested
        for i, line in enumerate(tqdm(lines[:5], desc=f"Processing {strategy_id}")):
            data = json.loads(line)
            human_msg = data["conversations"][1]["value"]
            ground_truth = data["conversations"][2]["value"]
            
            print(f"\n[ID: {strategy_id}] Processing query {i+1}/5...")
            
            try:
                response = llm.invoke(human_msg)
                model_answer = response.content
                print(f"✅ Success! Length: {len(model_answer)} chars.")
            except Exception as e:
                print(f"❌ Error invoking LLM for {strategy_id}: {e}")
                model_answer = "ERROR"

            results.append({
                "question_with_context": human_msg,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "strategy": strategy_id
            })
            
            # Incremental save
            with open(output_path, "w") as f_out:
                json.dump(results, f_out, indent=4)
    
    print(f"\n✅ Selesai {strategy_id}! Results: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch inference for Ekonomi Syariah Chatbot.")
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
