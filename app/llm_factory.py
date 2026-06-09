import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm(model_name=None, temperature=0.7):
    engine = os.getenv("INFERENCE_ENGINE", "ollama").lower()
    
    if not model_name:
        model_name = os.getenv("MODEL_NAME", "hanif-raft-v3:latest")

    if engine == "vllm":
        base_url = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        print(f"[INFO] Using vLLM engine at {base_url}")
        return ChatOpenAI(
            model=model_name,
            openai_api_key="EMPTY",
            openai_api_base=base_url,
            temperature=temperature
        )
    elif engine == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
        print(f"[INFO] Using OpenRouter engine")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            model_kwargs={
                "extra_headers": {
                    "HTTP-Referer": "https://github.com/alif-faturrohman/ekonomi-syariah-chatbot-llm",
                    "X-Title": "HANIF Bot"
                }
            }
        )
    else:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"[INFO] Using Ollama engine at {base_url}")
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url
        )
