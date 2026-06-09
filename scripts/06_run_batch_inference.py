import json
import os
import argparse
from langchain_ollama import ChatOllama
from tqdm import tqdm

# --- CONFIG ---
STRATEGIES = {
    "strat1": "data/dataset_splits/strat1_pure_aug_test.jsonl",
    "strat2": "data/dataset_splits/strat2_cross_test.jsonl",
    "strat3": "data/dataset_splits/strat3_dual_test.jsonl",
}
OUTPUT_DIR = "eval_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_inference(strategy_name):
    dataset_path = STRATEGIES.get(strategy_name)
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"⚠️ Dataset untuk {strategy_name} tidak ditemukan di {dataset_path}")
        return

    model_name = f"hanif_{strategy_name}:latest"
    print(f"\n🚀 Memulai Inference: {strategy_name.upper()}")
    print(f"🤖 Model: {model_name}")
    print(f"📂 Dataset: {dataset_path}")
    
    llm = ChatOllama(model=model_name, temperature=0.0)

    results = []
    output_path = os.path.join(OUTPUT_DIR, f"{strategy_name}_results.json")
    
    with open(dataset_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines[:1], desc=f"Processing {strategy_name}")):
            data = json.loads(line)
            human_msg = data["conversations"][1]["value"]
            ground_truth = data["conversations"][2]["value"]
            
            print(f"\n[Strategy: {strategy_name}] Processing query {i+1}/5...")
            
            try:
                response = llm.invoke(human_msg)
                model_answer = response.content
                print(f"✅ Success! Generated {len(model_answer)} characters.")
            except Exception as e:
                print(f"❌ Error invoking LLM for {strategy_name}: {e}")
                model_answer = "ERROR"

            results.append({
                "question_with_context": human_msg,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "strategy": strategy_name
            })
            
            # Save incrementally after each query
            with open(output_path, "w") as f_out:
                json.dump(results, f_out, indent=4)
    
    print(f"\n✅ Selesai strategi {strategy_name}! Hasil akhir di: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Inference script for model evaluation.")
    parser.add_argument(
        "--strat", 
        nargs="+", 
        choices=["strat1", "strat2", "strat3"], 
        default=["strat1", "strat2", "strat3"],
        help="List strategi yang ingin di-test (contoh: --strat strat1 strat2)"
    )
    
    args = parser.parse_args()
    
    for strategy in args.strat:
        run_inference(strategy)

if __name__ == "__main__":
    main()
