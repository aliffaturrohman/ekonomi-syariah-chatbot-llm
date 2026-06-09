import json
import os
from langchain_ollama import ChatOllama
from tqdm import tqdm

STRATEGIES = ["strat1", "strat2", "strat3"]
DATASET_DIR = "data/dataset_splits"
OUTPUT_DIR = "eval_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_inference(strategy_name):
    dataset_path = os.path.join(DATASET_DIR, f"{strategy_name}_test.jsonl")
    if strategy_name == "strat1":
        dataset_path = os.path.join(DATASET_DIR, "strat1_pure_aug_test.jsonl")
    elif strategy_name == "strat2":
        dataset_path = os.path.join(DATASET_DIR, "strat2_cross_test.jsonl")
    elif strategy_name == "strat3":
        dataset_path = os.path.join(DATASET_DIR, "strat3_dual_test.jsonl")

    if not os.path.exists(dataset_path):
        print(f"⚠️ Dataset not found: {dataset_path}")
        return

    model_name = f"hanif_{strategy_name}:latest"
    print(f"🚀 Running inference for {strategy_name} using {model_name}")
    
    llm = ChatOllama(model=model_name, temperature=0.0) # Use 0.0 for evaluation consistency

    results = []
    
    with open(dataset_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Processing {strategy_name}"):
            data = json.loads(line)
            # conversations[1] is human (context + question)
            # conversations[2] is gpt (ground truth)
            
            human_msg = data["conversations"][1]["value"]
            ground_truth = data["conversations"][2]["value"]
            
            # Split context and question if possible
            # In our dataset, human_msg usually looks like "Konteks:\n...\n\nPertanyaan:\n..."
            # But let's just pass the whole thing as user input.
            
            try:
                response = llm.invoke(human_msg)
                model_answer = response.content
            except Exception as e:
                print(f"Error invoking LLM: {e}")
                model_answer = "ERROR"

            results.append({
                "question_with_context": human_msg,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "strategy": strategy_name
            })

    output_path = os.path.join(OUTPUT_DIR, f"{strategy_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Saved results to {output_path}")

def main():
    for strat in STRATEGIES:
        run_inference(strat)

if __name__ == "__main__":
    main()
