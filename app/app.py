import streamlit as st
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


DB_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
MODEL_NAME = "hanif-raft-v3:latest"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}
    )
    
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.2,
        base_url="http://localhost:11434"
    )
    
    return retriever, llm

retriever, llm = load_resources()

def parse_output(text):
    """
    Fungsi ini mencari tag <thought>...</thought>
    Output: (thought_content, answer_content)
    """
    pattern = r"<thought>(.*?)</thought>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        thought = match.group(1).strip()
        answer = match.group(2).strip()
        return thought, answer
    else:
        return None, text.strip()

system_prompt = """
Kamu adalah HANIF (Helpful AI for Noble Islamic Finance).
Tugasmu adalah menjawab pertanyaan pengguna berdasarkan konteks dokumen yang diberikan.

ATURAN FORMAT WAJIB:
1. Mulailah dengan analisis mendalam menggunakan tag <thought>.
2. Jelaskan dokumen mana yang relevan dan kenapa.
3. Setelah tag penutup </thought>, berikan jawaban akhir yang sopan dan jelas kepada user.
4. Tidak perlu menyebutkan menurut teks, menurut dokumen, atau sebagainya yang berhubungan dengan pertanyaan user.
5. Pastikan jawaban spesifik dan tidak bertele-tele dalam menjawab pertanyaan.

JANGAN PERNAH MENGHILANGKAN TAG <thought>.
"""

prompt = ChatPromptTemplate.from_template(
    """
    {system_prompt}

    Konteks Referensi:
    {context}

    Pertanyaan User: 
    {question}
    """
)

def format_docs(docs):
    return "\n\n---\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "system_prompt": lambda x: system_prompt}
    | prompt
    | llm
    | StrOutputParser()
)


st.set_page_config(page_title="HANIF - Ekonomi Syariah Bot", page_icon="🕌")
st.title("🕌 HANIF: Asisten Ekonomi Syariah")
st.caption("Bertanya seputar hukum ekonomi Islam berdasarkan literatur kitab.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("thought"):
            with st.expander("🔍 Lihat Proses Berpikir (Chain of Thought)"):
                st.write(message["thought"])
        st.markdown(message["content"])

if prompt_input := st.chat_input("Tanyakan sesuatu (misal: Apa hukum riba?)"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Sedang membuka kitab dan berpikir..."):
            try:
                raw_response = rag_chain.invoke(prompt_input)
                
                thought, answer = parse_output(raw_response)
                
                if thought:
                    with st.expander("🔍 Lihat Proses Berpikir (Chain of Thought)"):
                        st.markdown(thought)
                
                st.markdown(answer)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "thought": thought
                })
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")