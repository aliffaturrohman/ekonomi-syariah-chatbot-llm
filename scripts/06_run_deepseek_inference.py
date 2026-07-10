"""DeepSeek v4 Flash inference for baseline comparison.
Runs strat1/2/3 test datasets through DeepSeek v4 Flash API.
Output matches format expected by BLEU/ROUGE and RAGAS eval scripts.
"""
import json
import os
import time
import sys
from dotenv import load_dotenv

# Ensure we can find the .env in the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
load_dotenv(os.path.join(project_root, ".env"))

from langchain_openai import ChatOpenAI
from tqdm import tqdm

DEEPSEEK_MODEL = "deepseek-chat"
INPUT_DIR = os.path.join(project_root, "data/dataset_splits")
OUTPUT_DIR = os.path.join(project_root, "eval_results_full")

DATASET_MAP = {
    "strat1": "strat1_pure_aug_test.jsonl",
    "strat2": "strat2_cross_test.jsonl",
    "strat3": "strat3_dual_test.jsonl",
}


def run_deepseek_inference(strategy: str):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not set in .env!")
        return

    dataset_file = DATASET_MAP.get(strategy)
    dataset_path = os.path.join(INPUT_DIR, dataset_file)
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return

    metadata_tag = f"deepseek_v4_flash_{strategy}"
    output_path = os.path.join(OUTPUT_DIR, f"results_{metadata_tag}.json")

    print(f"\n{'='*60}")
    print(f"🚀 DeepSeek v4 Flash Inference: {strategy.upper()}")
    print(f"📂 Dataset: {dataset_file}")
    print(f"💾 Output: {os.path.basename(output_path)}")
    print(f"{'='*60}\n")

    llm = ChatOpenAI(
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=api_key,
        model_name=DEEPSEEK_MODEL,
        temperature=0.0,
        max_tokens=4096,
    )

    # Resume logic
    results = []
    processed_indices = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                results = json.load(f)
                processed_indices = {item["query_index"] for item in results}
            print(f"⏩ Resuming: {len(processed_indices)} already processed.")
        except Exception:
            print("⚠️ Corrupted output, starting fresh.")
            results = []

    with open(dataset_path, "r") as f:
        lines = f.readlines()

    total = len(lines)
    remaining = total - len(processed_indices)
    print(f"📊 Total: {total} queries ({remaining} remaining)\n")

    if remaining == 0:
        print("✅ All queries already processed!")
        return

    for i in tqdm(range(total), desc=f"DeepSeek {strategy}"):
        query_idx = i + 1
        if query_idx in processed_indices:
            continue

        data = json.loads(lines[i])
        human_msg = data["conversations"][1]["value"]
        ground_truth = data["conversations"][2]["value"]

        # Retry with exponential backoff
        model_answer = None
        for attempt in range(7):
            try:
                response = llm.invoke(human_msg)
                model_answer = response.content
                break
            except Exception as e:
                wait = min(2 ** attempt, 60)
                tqdm.write(f"\n⚠️ Query {query_idx} attempt {attempt+1}/7: {e}")
                if attempt < 6:
                    tqdm.write(f"   Retry in {wait}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"❌ Giving up on query {query_idx}")

        if model_answer is None:
            model_answer = "ERROR"

        results.append({
            "query_index": query_idx,
            "question_with_context": human_msg,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "model_used": f"deepseek/{DEEPSEEK_MODEL}",
            "metadata": metadata_tag,
        })

        # Save every 10 items
        if query_idx % 10 == 0:
            with open(output_path, "w") as f_out:
                json.dump(results, f_out, indent=2)

    with open(output_path, "w") as f_out:
        json.dump(results, f_out, indent=2)

    print(f"\n✅ {strategy} done! {len(results)} results at {os.path.basename(output_path)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strat", nargs="+", choices=["strat1", "strat2", "strat3"],
                        default=["strat1", "strat2", "strat3"])
    args = parser.parse_args()

    for s in args.strat:
        run_deepseek_inference(s)


if __name__ == "__main__":
    main()
