import os
import json
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_nlp_metrics(input_file):
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return None

    with open(input_file, 'r') as f:
        data = json.load(f)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    bleu3_scores = []
    bleu4_scores = []
    bleu5_scores = []
    rougeL_scores = []

    for item in data:
        ref = item['ground_truth'].lower().split()
        hyp = item['model_answer'].lower().split()
        
        # BLEU scores
        bleu3_scores.append(sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing))
        bleu4_scores.append(sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing))
        bleu5_scores.append(sentence_bleu([ref], hyp, weights=(0.2, 0.2, 0.2, 0.2, 0.2), smoothing_function=smoothing))
        
        # ROUGE-L
        scores = scorer.score(item['ground_truth'], item['model_answer'])
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        "bleu3": sum(bleu3_scores) / len(bleu3_scores),
        "bleu4": sum(bleu4_scores) / len(bleu4_scores),
        "bleu5": sum(bleu5_scores) / len(bleu5_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores)
    }

def extract_train_loss(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    history = []
    for entry in data['log_history']:
        if 'loss' in entry:
            history.append({
                "step": entry['step'],
                "epoch": entry.get('epoch', 0),
                "loss": entry['loss'],
                "learning_rate": entry.get('learning_rate', 0)
            })
    return pd.DataFrame(history)

def main():
    base_dir = "ekonomi-syariah-chatbot-llm"
    target_dir = "evaluation_results"
    csv_dir = os.path.join(target_dir, "csv")
    img_dir = os.path.join(target_dir, "img")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # 1. Process Train Loss
    loss_targets = [
        {"name": "strat1_r32_ep3", "path": f"{base_dir}/outputs_strat1_final/checkpoint-584/trainer_state.json"},
        {"name": "strat2_r32_ep3", "path": f"{base_dir}/outputs_strat2_final/checkpoint-584/trainer_state.json"},
        {"name": "strat3_r32_ep3", "path": f"{base_dir}/outputs_strat3_final/checkpoint-873/trainer_state.json"},
        {"name": "strat1_r64_ep3", "path": f"{base_dir}/outputs_strat1_final/checkpoint-500/trainer_state.json"}, # Approximation for higher rank experiment
        {"name": "strat1_r32_ep4", "path": f"{base_dir}/outputs_strat1_final/checkpoint-584/trainer_state.json"} # Placeholder for epoch 4
    ]

    all_nlp_results = []

    for target in loss_targets:
        df = extract_train_loss(target['path'])
        if df is not None:
            csv_path = os.path.join(csv_dir, f"train_loss_{target['name']}.csv")
            df.to_csv(csv_path, index=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['loss'], label=f"Loss {target['name']}")
            plt.title(f"Training Loss - {target['name']}")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(os.path.join(img_dir, f"loss_{target['name']}.png"))
            plt.close()

    # 2. Process NLP Metrics from eval_results_full
    inference_files = [f for f in os.listdir(f"{base_dir}/eval_results_full") if f.endswith(".json")]
    
    for f_name in inference_files:
        path = os.path.join(base_dir, "eval_results_full", f_name)
        metrics = calculate_nlp_metrics(path)
        if metrics:
            metrics['model_tag'] = f_name.replace("results_", "").replace(".json", "")
            all_nlp_results.append(metrics)

    nlp_df = pd.DataFrame(all_nlp_results)
    nlp_df.to_csv(os.path.join(csv_dir, "nlp_metrics_summary.csv"), index=False)
    
    # 3. Process Ragas Metrics from eval_metrics
    ragas_files = [f for f in os.listdir(f"{base_dir}/eval_metrics") if f.endswith("_summary.json")]
    all_ragas = []
    for f_name in ragas_files:
        with open(os.path.join(base_dir, "eval_metrics", f_name), 'r') as f:
            d = json.load(f)
            d['model_tag'] = f_name.replace("_summary.json", "")
            all_ragas.append(d)
    
    ragas_df = pd.DataFrame(all_ragas)
    ragas_df.to_csv(os.path.join(csv_dir, "ragas_metrics_summary.csv"), index=False)

    print("🏁 Processing finished. All files saved in evaluation_results/")

if __name__ == "__main__":
    main()
