import json
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Auto-apply serialization patch for Python 3.14 compatibility
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(script_dir)
    from patch_datasets import patch_datasets_dill
    patch_datasets_dill()
except Exception as e:
    print(f"Warning: Auto-patcher failed: {e}")

import sys
import types
import pickle

# Patch pickle.Pickler.save_dict to workaround dill's _batch_setitems incompatibility in Python 3.14
if hasattr(pickle.Pickler, "save_dict"):
    original_save_dict = pickle.Pickler.save_dict
    def patched_save_dict(self, obj):
        try:
            return original_save_dict(self, obj)
        except TypeError as e:
            if "_batch_setitems" in str(e):
                self.write(pickle.DICT)
                self.memoize(obj)
                self._batch_setitems(iter(obj.items()))
                return
            raise
    pickle.Pickler.save_dict = patched_save_dict

if "langchain_community" not in sys.modules:
    import langchain_community
if not hasattr(langchain_community, "chat_models"):
    langchain_community.chat_models = types.ModuleType("langchain_community.chat_models")
    sys.modules["langchain_community.chat_models"] = langchain_community.chat_models
if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertexai_mock = types.ModuleType("langchain_community.chat_models.vertexai")
    vertexai_mock.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_mock

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas.run_config import RunConfig

# --- CONFIG ---
INPUT_DIR = "eval_results_full"
OUTPUT_DIR = "eval_metrics"
OLLAMA_MODEL = "qwen2.5:7b" 
HF_EMBED_MODEL = "intfloat/multilingual-e5-large"
OPENROUTER_MODEL = "deepseek/deepseek-v4-flash" # Specified model for OpenRouter evaluation

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def extract_context_and_question(text):
    # Standard format: "Konteks:\n{context}\n\nPertanyaan:\n{question}"
    try:
        if "Konteks:" in text and "Pertanyaan:" in text:
            parts = text.split("Pertanyaan:")
            question = parts[1].strip()
            context_part = parts[0].replace("Konteks:", "").strip()
            return context_part, question
        else:
            return "", text
    except:
        return "", text

def run_ragas_eval(strategy_name, metadata_tag, judge_type):
    input_path = os.path.join(INPUT_DIR, f"results_{metadata_tag}.json")
    if not os.path.exists(input_path):
        print(f"⚠️ Result file not found: {input_path}")
        return

    # Determine judge and naming
    if judge_type == "openrouter":
        judge_name_tag = "openrouter_deepseek_v4_flash"
        print(f"🤖 Initializing OpenRouter Judge ({OPENROUTER_MODEL})...")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ Error: OPENROUTER_API_KEY environment variable not set!")
            print("👉 Silakan jalankan: export OPENROUTER_API_KEY='your-api-key' terlebih dahulu.")
            return
        judge_llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            model_name=OPENROUTER_MODEL,
        )
    elif judge_type == "deepseek":
        judge_name_tag = "deepseek_api"
        print(f"🤖 Initializing DeepSeek API Judge (deepseek-chat)...")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ Error: DEEPSEEK_API_KEY environment variable not set!")
            print("👉 Silakan set DEEPSEEK_API_KEY di dalam file .env atau env var Anda.")
            return
        judge_llm = ChatOpenAI(
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=api_key,
            model_name="deepseek-chat",
        )
    else:
        judge_name_tag = "ollama_qwen"
        print(f"🤖 Initializing Ollama Judge ({OLLAMA_MODEL})...")
        judge_llm = ChatOllama(model=OLLAMA_MODEL)

    # Gunakan HuggingFace Embeddings agar sama dengan yang digunakan saat ingest data
    print(f"🧠 Initializing HuggingFace Embeddings ({HF_EMBED_MODEL})...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBED_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"\n" + "="*50)
    print(f"📊 Evaluating {strategy_name} with Ragas (Judge: {judge_name_tag})...")
    print(f"📄 File: {input_path}")
    print("="*50)
    
    with open(input_path, "r") as f:
        data = json.load(f)

    output_csv = os.path.join(OUTPUT_DIR, f"{strategy_name}_{metadata_tag}_{judge_name_tag}_metrics.csv")
    output_json = os.path.join(OUTPUT_DIR, f"{strategy_name}_{metadata_tag}_{judge_name_tag}_summary.json")

    processed_indices = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'query_index' in existing_df.columns:
                processed_indices = set(existing_df['query_index'].tolist())
            print(f"⏩ Resuming: {len(processed_indices)} queries already evaluated.")
        except Exception as e:
            print(f"⚠️ Could not read existing results, starting fresh: {e}")

    processed_data = {
        "query_index": [],
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for idx, item in enumerate(data):
        query_index = item.get("query_index", idx + 1)
        context, question = extract_context_and_question(item["question_with_context"])
        processed_data["query_index"].append(query_index)
        processed_data["question"].append(question)
        processed_data["answer"].append(item["model_answer"])
        processed_data["contexts"].append([context])
        processed_data["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(processed_data)

    # Run evaluation in mini-batches for both parallelism and incremental safety
    batch_size = 20
    for i in range(0, len(dataset), batch_size):
        batch_indices = []
        for j in range(i, min(i + batch_size, len(dataset))):
            if dataset[j]["query_index"] not in processed_indices:
                batch_indices.append(j)
        
        if not batch_indices:
            continue
            
        mini_batch_dataset = dataset.select(batch_indices)
        current_query_indices = [dataset[idx]["query_index"] for idx in batch_indices]
        
        print(f"🚀 Evaluating batch: queries {current_query_indices[0]} to {current_query_indices[-1]} ({len(mini_batch_dataset)} rows)...")
        
        try:
            # Set higher max_workers for API parallel calls
            max_workers = 15 if judge_type == "openrouter" else 4
            
            result = evaluate(
                mini_batch_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=judge_llm,
                embeddings=embeddings,
                run_config=RunConfig(max_workers=max_workers),
            )
            
            df_batch = result.to_pandas()
            # Ensure query_index is preserved correctly
            df_batch.insert(0, 'query_index', current_query_indices)
            
            # Save incrementally
            if not os.path.exists(output_csv):
                df_batch.to_csv(output_csv, index=False)
            else:
                df_batch.to_csv(output_csv, mode='a', header=False, index=False)
            
            print(f"✅ Batch completed and saved.")
        except Exception as e:
            print(f"❌ Error evaluating batch starting at {current_query_indices[0]}: {e}")

    # Generate summary after all rows are done
    try:
        final_df = pd.read_csv(output_csv)
        summary = final_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean().to_dict()
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=4, default=str)
        print(f"✅ Summary saved to {output_json}")
    except Exception as e:
        print(f"⚠️ Could not generate summary: {e}")

    print(f"✅ Evaluation done for {strategy_name}.")
    print(f"📈 Results saved to:\n  - {output_csv}\n  - {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Ragas Evaluation Script with Ollama/OpenRouter/DeepSeek support")
    parser.add_argument("--judge", choices=["ollama", "openrouter", "deepseek"], default="ollama", help="Choose the judge LLM (ollama, openrouter, or deepseek)")
    args = parser.parse_args()

    # Load completed training parameters from parameter_end_training.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    completed_path = os.path.join(script_dir, "parameter_end_training.json")
    
    completed_jobs = []
    if os.path.exists(completed_path):
        try:
            with open(completed_path, "r") as f:
                completed_jobs = json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading parameter_end_training.json: {e}")

    # Fallback/Historical mapping for jobs without param_tag
    HISTORICAL_MAP = {
        ("strat1", 5e-05, 2, 16): "strat1_lr5e5_ep2_20260602_1659",
        ("strat2", 5e-05, 2, 16): "strat2_lr5e5_ep2_20260602_1802",
        ("strat3", 5e-05, 2, 16): "strat3_lr5e5_ep2_20260604_0943",
    }

    jobs_to_evaluate = []
    skipped_inference_count = 0
    already_evaluated_count = 0

    for job in completed_jobs:
        if not job.get("status", "").startswith("Sukses"):
            continue

        strategy = job.get("strategy")
        lr = job.get("lr")
        epochs = job.get("epochs")
        rank = job.get("rank")
        
        # Get metadata tag
        metadata_tag = job.get("param_tag")
        if not metadata_tag:
            # Check historical map
            metadata_tag = HISTORICAL_MAP.get((strategy, lr, epochs, rank))
            
        if not metadata_tag:
            # Fallback for old ones without metadata (construct a guess)
            metadata_tag = f"{strategy}_final_r{rank}_lr2e5_ep{epochs}"

        # 1. Check if inference file exists in INPUT_DIR ("eval_results_full")
        input_filename = f"results_{metadata_tag}.json"
        input_path = os.path.join(INPUT_DIR, input_filename)
        
        if not os.path.exists(input_path):
            # If the inference file is not found, we skip it because it hasn't been run yet
            skipped_inference_count += 1
            continue

        # 2. Check if the evaluation output summary file already exists
        # Determine judge name tag (mirrors logic in run_ragas_eval)
        if args.judge == "openrouter":
            judge_name_tag = "openrouter_deepseek_v4_flash"
        elif args.judge == "deepseek":
            judge_name_tag = "deepseek_api"
        else:
            judge_name_tag = "ollama_qwen"
            
        summary_path = os.path.join(OUTPUT_DIR, f"{strategy}_{metadata_tag}_{judge_name_tag}_summary.json")
        if os.path.exists(summary_path):
            already_evaluated_count += 1
            continue

        # If it passed both checks, add it to the execution list!
        jobs_to_evaluate.append((strategy, metadata_tag))

    if not jobs_to_evaluate:
        print("\n" + "="*60)
        print("🔔 NOTIFIKASI: Semua evaluasi model sudah dijalankan!")
        if skipped_inference_count > 0:
            print(f"ℹ️  Ada {skipped_inference_count} model sukses ditraining tetapi file hasil inference-nya")
            print(f"    belum ada di folder '{INPUT_DIR}/'. Silakan jalankan inference terlebih dahulu.")
        if already_evaluated_count > 0:
            print(f"ℹ️  {already_evaluated_count} model telah memiliki hasil evaluasi lengkap di '{OUTPUT_DIR}/'.")
        print("="*60 + "\n")
        return

    print(f"\n📊 Ditemukan {len(jobs_to_evaluate)} model baru yang siap dievaluasi.")
    for strategy, metadata_tag in jobs_to_evaluate:
        run_ragas_eval(strategy, metadata_tag, args.judge)

if __name__ == "__main__":
    main()
