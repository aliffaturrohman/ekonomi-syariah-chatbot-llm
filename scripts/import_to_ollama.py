import os
import subprocess

STRATEGIES = ["strat1", "strat2", "strat3"]

def import_to_ollama(strategy_name):
    # Use absolute path for GGUF
    project_root = os.getcwd()
    adapter_dir = f"models/adapters/qwen_raft_ekonomi_syariah_{strategy_name}_gguf"
    gguf_path = os.path.join(project_root, adapter_dir, "Qwen2.5-7B-Instruct.Q4_K_M.gguf")

    if not os.path.exists(gguf_path):
        print(f"⚠️ GGUF file not found for {strategy_name} at {gguf_path}")
        return

    # Use underscore instead of hyphen just in case
    model_name = f"hanif_{strategy_name}:latest"
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
    modelfile_path = os.path.join(project_root, adapter_dir, f"ModelFile_{strategy_name}")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"🚀 Creating Ollama model: {model_name}")
    subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], check=True)
    print(f"✅ Created {model_name}")

def main():
    for strat in STRATEGIES:
        import_to_ollama(strat)

if __name__ == "__main__":
    main()
