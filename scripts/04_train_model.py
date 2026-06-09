import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

# Auto-apply serialization patch for Python 3.14 compatibility
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(script_dir)
    from patch_datasets import patch_datasets_dill
    patch_datasets_dill()
except Exception as e:
    print(f"Warning: Auto-patcher failed: {e}")

import ctypes
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.abspath(os.path.join(script_dir, "../venv"))
cuda_lib_path = "/home/alif-faturrohman/coding/ekonomi-syariah-chatbot-llm/venv/lib/python3.14/site-packages/nvidia/cu13/lib/libnvJitLink.so.13"
if os.path.exists(cuda_lib_path):
    try:
        ctypes.CDLL(cuda_lib_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Warning: Could not pre-load CUDA library: {e}")

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import argparse
import sys
from datetime import datetime
import pandas as pd
import json

STRATEGIES = {
    "strat1": os.path.join(script_dir, "../data/dataset_splits/strat1_pure_aug_train.jsonl"),
    "strat2": os.path.join(script_dir, "../data/dataset_splits/strat2_cross_train.jsonl"),
    "strat3": os.path.join(script_dir, "../data/dataset_splits/strat3_dual_train.jsonl"),
}

MAX_SEQ_LENGTH = 2048
DTYPE = None 
LOAD_IN_4BIT = True 

def run_training(strategy_name, lr_val, epochs, rank):
    dataset_path = STRATEGIES.get(strategy_name)
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"⚠️  Dataset not found: {dataset_path}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    param_tag = f"final_r{rank}_lr2e5_ep{epochs}_{timestamp}"
    output_dir = os.path.abspath(os.path.join(script_dir, f"../models/adapters/qwen_raft_ekonomi_syariah_{strategy_name}_{param_tag}"))
    log_dir = os.path.join(script_dir, f"logs/{strategy_name}_{param_tag}")
    
    # Dynamic batch size based on GPU VRAM detection
    batch_size = 1
    grad_accum = 8
    
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️  Detected GPU VRAM: {total_vram_gb:.2f} GB")
        
        if total_vram_gb <= 13.0: # ~12GB (e.g. RTX 3060/4060 Ti 12GB)
            batch_size = 1
            grad_accum = 8
        elif total_vram_gb <= 18.0: # ~16GB (e.g. T4, RTX 4080 16GB, RTX 4060 Ti 16GB)
            batch_size = 2
            grad_accum = 4
        elif total_vram_gb <= 26.0: # ~24GB/25GB (e.g. RTX 3090/4090, A10G)
            batch_size = 4
            grad_accum = 2
        else: # > 26GB (e.g. A40, A6000, A100 40GB/80GB)
            batch_size = 8
            grad_accum = 1
    else:
        print("⚠️ No CUDA device detected! Defaulting to Batch Size: 1, Accumulation: 8")
        
    print(f"\n" + "="*50)
    print(f"🚀 FINAL TRAINING: {strategy_name.upper()}")
    print(f"⚙️  RANK: {rank}, LR: {lr_val}, EPOCHS: {epochs}")
    print(f"📊 BATCH SIZE: {batch_size}, ACCUMULATION STEPS: {grad_accum} (Effective: {batch_size * grad_accum})")
    print(f"💾 OUTPUT: {output_dir}")
    print(f"📈 Logs: {log_dir}")
    print("="*50 + "\n")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = rank, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = rank * 2,
        lora_dropout = 0, 
        bias = "none", 
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "chatml",
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    # Keep num_proc=None for safety despite library patch
    dataset = dataset.map(formatting_prompts_func, batched = True, num_proc = None)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = None,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            num_train_epochs = epochs, 
            learning_rate = lr_val,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.1,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = f"outputs_{strategy_name}_final",
            logging_dir = log_dir,
            report_to = "tensorboard",
        ),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save Loss to CSV
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(f"{output_dir}/loss_history.csv", index=False)
    
    del model
    del tokenizer
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ Finished {strategy_name}. VRAM Cleared.\n")
    return param_tag, timestamp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true", help="Run training from queue JSON file")
    parser.add_argument("--train", nargs="+", default=["strat1", "strat2", "strat3"])
    args = parser.parse_args()
    
    if args.queue:
        queue_path = os.path.join(script_dir, "training_queue.json")
        completed_path = os.path.join(script_dir, "parameter_end_training.json")
        
        if not os.path.exists(queue_path):
            print(f"⚠️ Queue file not found at: {queue_path}")
            with open(queue_path, "w") as f:
                json.dump([], f, indent=4)
            return
            
        with open(queue_path, "r") as f:
            queue = json.load(f)
            
        if not queue:
            print("🎉 Queue is empty! No models to train.")
            return
            
        while len(queue) > 0:
            job = queue[0]
            strategy = job.get("strategy")
            lr = job.get("lr")
            epochs = job.get("epochs")
            rank = job.get("rank")
            
            print(f"\n==========================================")
            print(f"Executing queued job: Strategy={strategy}, LR={lr}, Epochs={epochs}, Rank={rank}")
            print(f"==========================================\n")
            
            success = False
            error_msg = ""
            p_tag = ""
            t_stamp = ""
            try:
                p_tag, t_stamp = run_training(strategy, lr_val=lr, epochs=epochs, rank=rank)
                success = True
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error during training of {strategy}: {error_msg}")
                print("Recording failure and proceeding to next job in queue...")
                
            completed_jobs = []
            if os.path.exists(completed_path):
                with open(completed_path, "r") as f:
                    try:
                        completed_jobs = json.load(f)
                    except json.JSONDecodeError:
                        completed_jobs = []
                        
            job["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if success:
                job["status"] = "Sukses"
                job["param_tag"] = p_tag
                job["timestamp"] = t_stamp
            else:
                job["status"] = f"Error: {error_msg}"
            completed_jobs.append(job)
            
            with open(completed_path, "w") as f:
                json.dump(completed_jobs, f, indent=4)
                
            queue.pop(0)
            with open(queue_path, "w") as f:
                json.dump(queue, f, indent=4)
    else:
        for s in args.train:
            run_training(s, lr_val=2e-5, epochs=3, rank=32)

if __name__ == "__main__":
    main()
