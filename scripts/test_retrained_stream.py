import sys
import time
from langchain_ollama import ChatOllama

def test_retrained_models():
    models = ["hanif_strat1_retrain:latest", "hanif_strat2_retrain:latest"]
    prompt = "Halo, perkenalkan dirimu secara singkat."
    
    for model_name in models:
        print(f"\n" + "="*50)
        print(f"🚀 TESTING RETRAINED MODEL: {model_name}")
        print(f"❓ Prompt: {prompt}")
        print("="*50)
        print("--- RESPONSE STREAM ---")
        
        llm = ChatOllama(model=model_name, temperature=0.7)
        
        start_time = time.time()
        try:
            for chunk in llm.stream(prompt):
                print(chunk.content, end="", flush=True)
                if time.time() - start_time > 30:
                    print("\n\n[Timeout 30s reached]")
                    break
        except Exception as e:
            print(f"\n\n❌ Error pada {model_name}: {e}")
        
        print("\n--- END OF MODEL ---")
        time.sleep(3)

if __name__ == "__main__":
    test_retrained_models()
