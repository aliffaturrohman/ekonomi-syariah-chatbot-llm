"""Evaluate BLEU & ROUGE on ALL inference results including DeepSeek baseline.
Writes comparison to finding.md-compatible format.
"""
import json
import os
import sys
import glob

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Ensure NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
eval_dir = os.path.join(project_root, "eval_results_full")
output_dir = os.path.join(project_root, "eval_metrics")
os.makedirs(output_dir, exist_ok=True)

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
smoothing = SmoothingFunction().method1


def calc_metrics(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    bleu3, bleu4, bleu5, rougeL = [], [], [], []
    n = len(data)

    for item in data:
        ref = item["ground_truth"].lower().split()
        hyp = item["model_answer"].lower().split()

        bleu3.append(sentence_bleu([ref], hyp, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing))
        bleu4.append(sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing))
        bleu5.append(sentence_bleu([ref], hyp, weights=(0.2, 0.2, 0.2, 0.2, 0.2), smoothing_function=smoothing))

        s = scorer.score(item["ground_truth"], item["model_answer"])
        rougeL.append(s["rougeL"].fmeasure)

    return {
        "n_samples": n,
        "bleu3": round(sum(bleu3) / n, 4),
        "bleu4": round(sum(bleu4) / n, 4),
        "bleu5": round(sum(bleu5) / n, 4),
        "rougeL": round(sum(rougeL) / n, 4),
    }


def main():
    # Read all results_*.json files
    files = sorted(glob.glob(os.path.join(eval_dir, "results_*.json")))
    print(f"Found {len(files)} result files\n")

    rows = []
    for fpath in files:
        fname = os.path.basename(fpath).replace("results_", "").replace(".json", "")
        print(f"📊 Evaluating {fname}...")
        m = calc_metrics(fpath)
        rows.append({"model": fname, **m})
        print(f"   BLEU-3={m['bleu3']}, BLEU-4={m['bleu4']}, ROUGE-L={m['rougeL']}")

    # Print comparison table
    print("\n" + "="*90)
    print(f"{'Model':<55} {'BLEU-3':<8} {'BLEU-4':<8} {'BLEU-5':<8} {'ROUGE-L':<8} {'N':<6}")
    print("-"*90)
    for r in rows:
        name = r["model"][:54]
        print(f"{name:<55} {r['bleu3']:<8} {r['bleu4']:<8} {r['bleu5']:<8} {r['rougeL']:<8} {r['n_samples']:<6}")
    print("="*90)

    # Save JSON
    outpath = os.path.join(output_dir, "bleu_rouge_all_models.json")
    with open(outpath, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n✅ Saved to {outpath}")


if __name__ == "__main__":
    main()
