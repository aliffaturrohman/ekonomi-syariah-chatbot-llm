from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- KONFIGURASI (SESUAIKAN PATH INI) ---
# Arahkan ke file MASTER yang sudah digabung (Identity + RAFT)
DATASET_PATH = "../data/dataset_training_ver2/MASTER_RAFT_DATASET.jsonl" 

# Tempat menyimpan hasil training (LoRA Adapters)
OUTPUT_DIR = "../models/adapters/qwen_raft_ekonomi_syariah_ver2"

# Konfigurasi VRAM RTX 3060 (12GB)
MAX_SEQ_LENGTH = 2048 # Panjang konteks maksimal. Jika OOM, turunkan ke 1024
DTYPE = None # Auto detect (biasanya bfloat16 atau float16)
LOAD_IN_4BIT = True # WAJIB True agar muat di 12GB

def main():
    print("🚀 Memulai Proses Fine-Tuning dengan Unsloth...")
    print(f"📂 Dataset: {DATASET_PATH}")
    
    # 1. Load Model Dasar (Base Model)
    # Kita pakai Qwen 2.5 Instruct versi 4-bit biar ringan
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 2. Pasang Adapter LoRA
    # Kita hanya melatih sebagian kecil otak model (Adapter) agar cepat & hemat memori
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank: Angka 16 standar bagus. Bisa naik ke 32/64 tapi lebih berat.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none", 
        use_gradient_checkpointing = "unsloth", # Fitur hemat memori Unsloth
        random_state = 3407,
    )

    # 3. Load & Format Dataset
    print("📖 Mempersiapkan Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # Kita gunakan format ChatML (Standar Qwen)
    # Unsloth punya helper function untuk ini
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

    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 4. Konfigurasi Trainer
    print("⚙️  Setting Hyperparameters...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, 
        
        args = TrainingArguments(
            # --- PENGATURAN VRAM ---
            per_device_train_batch_size = 2, # Kalau Error OOM, ubah jadi 1
            gradient_accumulation_steps = 4, # Akumulasi gradien (kompensasi batch kecil)
            
            # --- DURASI TRAINING ---
            # 1 Epoch biasanya CUKUP untuk RAFT agar model paham format tanpa merusak ingatan lamanya.
            num_train_epochs = 1, 
            
            # --- OPTIMIZER ---
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1, # Lapor setiap 1 step
            optim = "adamw_8bit", # Optimizer hemat RAM
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    # 5. Info GPU Sebelum Mulai
    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"🧠 GPU Terdeteksi: {gpu_stats.name}")
    print(f"   VRAM Total: {round(gpu_stats.total_memory / 1024**3, 2)} GB")

    # 6. EKSEKUSI TRAINING
    print("\n🔥 TRAINING DIMULAI! (Ini akan memakan waktu)...")
    trainer_stats = trainer.train()

    # 7. Simpan Hasil
    print(f"\n💾 Menyimpan Model LoRA ke: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ TRAINING SELESAI!")
    print("   Langkah selanjutnya: Export ke GGUF agar bisa dipakai Ollama.")

if __name__ == "__main__":
    main()