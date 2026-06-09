import sys
import time
from langchain_ollama import ChatOllama

def test_stream():
    model_name = "hanif_strat1:latest"
    print(f"🚀 Testing streaming with model: {model_name}")
    
    llm = ChatOllama(model=model_name, temperature=0.7)
    
    prompt = "Halo, perkenalkan dirimu."
    print(f"❓ Prompt: {prompt}\n")
    print("--- RESPONSE ---")
    
    start_time = time.time()
    try:
        # Streaming response
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            if time.time() - start_time > 30: # limit to 30 seconds for safety
                print("\n\n[Timeout reached]")
                break
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
    
    print("\n--- END ---")

if __name__ == "__main__":
    test_stream()
