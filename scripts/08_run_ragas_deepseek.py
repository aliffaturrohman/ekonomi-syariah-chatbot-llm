"""RAGAS evaluation for DeepSeek v4 Flash results - standalone script."""
import os, sys, json, pickle, types, torch
import pandas as pd
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
load_dotenv(os.path.join(project_root, ".env"))

# Dill patch
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

# VertexAI mock
if "langchain_community" not in sys.modules:
    import langchain_community
if not hasattr(langchain_community, "chat_models"):
    langchain_community.chat_models = types.ModuleType("langchain_community.chat_models")
    sys.modules["langchain_community.chat_models"] = langchain_community.chat_models
if "langchain_community.chat_models.vertexai" not in sys.modules:
    vm = types.ModuleType("langchain_community.chat_models.vertexai")
    vm.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = vm

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from datasets import Dataset


def run_ragas_for(strategy):
    fpath = f"eval_results_full/results_deepseek_v4_flash_{strategy}.json"
    if not os.path.exists(fpath):
        print(f"❌ {fpath} not found")
        return

    with open(fpath) as f:
        data = json.load(f)

    print(f"\n📊 RAGAS Evaluating DeepSeek {strategy} ({len(data)} rows)...")

    processed = {"query_index": [], "question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in data:
        text = item["question_with_context"]
        if "Konteks:" in text and "Pertanyaan:" in text:
            parts = text.split("Pertanyaan:")
            question = parts[1].strip()
            context_part = parts[0].replace("Konteks:", "").strip()
        else:
            question = text
            context_part = ""
        processed["query_index"].append(item.get("query_index", len(processed["query_index"]) + 1))
        processed["question"].append(question)
        processed["answer"].append(item["model_answer"])
        processed["contexts"].append([context_part])
        processed["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(processed)

    # OpenRouter required — DeepSeek API doesn't support n>1 (needed by answer_relevancy)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required for RAGAS (answer_relevancy needs n>1)")
    judge_llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        model_name="deepseek/deepseek-v4-flash",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    out_csv = f"eval_metrics/deepseek_v4_flash_{strategy}_deepseek_api_metrics.csv"
    out_json = f"eval_metrics/deepseek_v4_flash_{strategy}_deepseek_api_summary.json"

    bs = 20
    for i in range(0, len(dataset), bs):
        batch = dataset.select(range(i, min(i + bs, len(dataset))))
        print(f"  Batch {i//bs + 1}/{(len(dataset)+bs-1)//bs} ({len(batch)} rows)...")
        try:
            result = evaluate(
                batch,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=judge_llm,
                embeddings=embeddings,
                run_config=RunConfig(max_workers=5),
            )
            df = result.to_pandas()
            df.insert(0, "query_index", [processed["query_index"][j] for j in range(i, min(i+bs, len(dataset)))])
            if i == 0:
                df.to_csv(out_csv, index=False)
            else:
                df.to_csv(out_csv, mode="a", header=False, index=False)
            print(f"    ✅ Batch done")
        except Exception as e:
            print(f"    ❌ Batch error: {e}")

    try:
        final_df = pd.read_csv(out_csv)
        summary = final_df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().to_dict()
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"   ✅ Summary: {summary}")
    except Exception as e:
        print(f"   ⚠️ Summary: {e}")


if __name__ == "__main__":
    for s in ["strat1", "strat2", "strat3"]:
        run_ragas_for(s)
    print("\n🏁 ALL RAGAS DONE")
