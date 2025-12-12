import re
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


DB_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
MODEL_NAME = "hanif-raft-v3:latest"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


# -------------------------------
# LOAD RESOURCES
# -------------------------------
def load_resources():
    print("[INFO] Loading embeddings & Chroma DB...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}
    )

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.2,
        base_url="http://localhost:11434"
    )

    return vectorstore, llm


# -------------------------------
# PARSE THOUGHT
# -------------------------------
def parse_output(text):
    pattern = r"<thought>(.*?)</thought>(.*)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return None, text.strip()


# -------------------------------
# MAIN RAG FUNCTION
# -------------------------------
def run_rag(query, vectorstore, llm):
    print("\n===========================================")
    print(f"[USER QUESTION] {query}")
    print("===========================================\n")

    # Retrieve with score
    retrieved = vectorstore.similarity_search_with_score(query, k=3)

    # Show retrieved docs
    print("🔍 Retrieved Documents:")
    formatted_context = ""
    for i, (doc, score) in enumerate(retrieved, start=1):
        print(f"\n---[DOC {i}]--- (score={score:.4f})")
        print(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

        formatted_context += f"[DOC {i}]\n{doc.page_content}\n\n"

    # Build system prompt
    system_prompt = """
You are HANIF (Helpful AI for Noble Islamic Finance).
Your task is to answer the users question based on the provided document context.
MANDATORY FORMAT RULES:
Begin with a deep analysis enclosed in <thought> tags.
Explain which documents are relevant and why.
After the closing </thought> tag, provide a final answer that is polite and clear for the user.
Do not mention phrases such as according to the text, according to the document, based on the document, or anything similar referring to the source.
Ensure the answer is specific and preferably detailed when responding to the question.
Do not say “I dont know” if the information is not in the documents — use your own knowledge.

NEVER REMOVE THE <thought> TAG.
    """.strip()

    # Build final prompt
    prompt_templ = ChatPromptTemplate.from_template(
        """
{system_prompt}

Konteks Referensi:
{context}

Pertanyaan User:
{question}
        """
    )

    prompt_str = prompt_templ.format(
        system_prompt=system_prompt,
        context=formatted_context,
        question=query
    )

    print("\n===============================")
    print("📌 SYSTEM PROMPT USED")
    print("===============================")
    print(system_prompt)

    print("\n===============================")
    print("📌 FINAL PROMPT SENT TO LLM")
    print("===============================")
    print(prompt_str)

    # Run LLM
    print("\n🧠 Running LLM...")
    output = llm.invoke(prompt_str)
    output = output.content if hasattr(output, "content") else output

    print("\n===============================")
    print("📌 RAW LLM OUTPUT")
    print("===============================")
    print(output)

    # Parse output
    thought, answer = parse_output(output)

    print("\n===============================")
    print("🧩 PARSED THINKING (<thought>)")
    print("===============================")
    print(thought)

    print("\n===============================")
    print("💬 FINAL ANSWER")
    print("===============================")
    print(answer)

    print("\n===========================================")


# -------------------------------
# CLI LOOP
# -------------------------------
def cli_loop():
    vectorstore, llm = load_resources()

    print("\n===== HANIF CLI MODE =====")
    print("Tanyakan apa saja. Ketik 'exit' untuk keluar.\n")

    while True:
        query = input("\n❓ Pertanyaan: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Keluar...")
            break

        run_rag(query, vectorstore, llm)


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    cli_loop()
