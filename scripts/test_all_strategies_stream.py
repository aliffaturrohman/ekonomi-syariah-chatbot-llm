import sys
import time
from langchain_ollama import ChatOllama

def test_stream_all():
    models = ["hanif_strat1:latest", "hanif_strat2:latest", "hanif_strat3:latest"]
    prompt = "Halo, perkenalkan dirimu secara singkat."
    
    for model_name in models:
        print(f"\n" + "="*50)
        print(f"🚀 TESTING MODEL: {model_name}")
        print(f"❓ Prompt: {prompt}")
        print("="*50)
        print("--- RESPONSE STREAM ---")
        
        llm = ChatOllama(model=model_name, temperature=0.7)
        
        start_time = time.time()
        try:
            for chunk in llm.stream(prompt):
                print(chunk.content, end="", flush=True)
                if time.time() - start_time > 20:
                    print("\n\n[Timeout 20s reached]")
                    break
        except Exception as e:
            print(f"\n\n❌ Error pada {model_name}: {e}")
        
        print("\n--- END OF MODEL ---")
        time.sleep(2) # Jeda antar model

if __name__ == "__main__":
    test_stream_all()
