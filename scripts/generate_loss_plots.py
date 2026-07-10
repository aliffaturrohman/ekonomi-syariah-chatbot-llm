import json
import matplotlib.pyplot as plt
import os

def plot_loss(json_path, title, save_name):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    steps = []
    losses = []
    
    for entry in data['log_history']:
        if 'loss' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.title(f'Training Loss Curve - {title}', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save image
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ Grafik berhasil disimpan sebagai: {save_name}")

# Daftar model yang ingin dibuat grafiknya
models = [
    {
        "path": "ekonomi-syariah-chatbot-llm/outputs_strat1_final/checkpoint-584/trainer_state.json",
        "title": "Strat 1 (Rank 32, Ep 3)",
        "output": "loss_strat1_r32_ep3.png"
    },
    {
        "path": "ekonomi-syariah-chatbot-llm/outputs_strat3_final/checkpoint-873/trainer_state.json",
        "title": "Strat 3 (Rank 32, Ep 3) - Retrain",
        "output": "loss_strat3_retrain.png"
    }
]

for m in models:
    plot_loss(m['path'], m['title'], m['output'])
