import os
import subprocess

# The directories we are currently exporting to
NEW_GGUF_DIRS = [
    ("strat1_retrain", "models/adapters/qwen_raft_ekonomi_syariah_strat1_lr5e5_ep2_20260602_1659_gguf_gguf"),
    ("strat2_retrain", "models/adapters/qwen_raft_ekonomi_syariah_strat2_lr5e5_ep2_20260602_1802_gguf_gguf")
]

def import_to_ollama(tag, adapter_dir):
    project_root = os.getcwd()
    gguf_path = os.path.join(project_root, adapter_dir, "Qwen2.5-7B-Instruct.Q4_K_M.gguf")
    
    if not os.path.exists(gguf_path):
        print(f"⚠️ GGUF file not found at {gguf_path}")
        return

    model_name = f"hanif_{tag}:latest"
    modelfile_content = f"""
FROM {gguf_path}
TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
PARAMETER temperature 0.0
PARAMETER num_ctx 8192
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
SYSTEM \"\"\"Anda adalah HANIF, asisten AI Ekonomi Syariah. Jawablah pertanyaan pengguna dengan akurat berdasarkan konteks yang diberikan. Mulailah dengan analisis mendalam menggunakan tag <thought>.\"\"\"
"""
    modelfile_path = os.path.join(project_root, adapter_dir, f"ModelFile_{tag}")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"🚀 Creating Ollama model: {model_name}")
    subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], check=True)
    print(f"✅ Created {model_name}")

def main():
    for tag, path in NEW_GGUF_DIRS:
        import_to_ollama(tag, path)

if __name__ == "__main__":
    main()
