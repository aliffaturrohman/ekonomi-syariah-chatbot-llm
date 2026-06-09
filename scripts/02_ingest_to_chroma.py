import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

# --- KONFIGURASI ---
INPUT_MD_DIR = "data/processed/markdown_converted_from_pdf"
DB_DIR = "vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

def prepare_embeddings():
    print(f"⏳ Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
    # Menggunakan CUDA jika tersedia agar proses ingest cepat
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    
    model_kwargs = {'device': device} 
    encode_kwargs = {'normalize_embeddings': True} # Penting untuk model E5
    
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def reset_database():
    if os.path.exists(DB_DIR):
        print(f"🗑️ Menghapus database lama di {DB_DIR}...")
        try:
            shutil.rmtree(DB_DIR)
            print("✅ Database lama berhasil dihapus.")
        except Exception as e:
            print(f"⚠️ Gagal menghapus database: {e}")
    else:
        print("ℹ️ Database belum ada, akan dibuat baru.")

def main():
    if not os.path.exists(INPUT_MD_DIR):
        print(f"❌ Folder input tidak ditemukan: {os.path.abspath(INPUT_MD_DIR)}")
        return

    reset_database()
    embeddings = prepare_embeddings()
    
    # Setup Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "]
    )

    all_documents = []
    file_list = [f for f in os.listdir(INPUT_MD_DIR) if f.endswith(".md")]
    
    if not file_list:
        print(f"⚠️ Tidak ada file .md ditemukan di {INPUT_MD_DIR}")
        return

    print(f"📄 Membaca {len(file_list)} file markdown...")
    for filename in tqdm(file_list, desc="Reading Files"):
        file_path = os.path.join(INPUT_MD_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": filename, "chunk_id": i}
                    )
                )
        except Exception as e:
            print(f"❌ Gagal membaca {filename}: {e}")

    if not all_documents:
        print("⚠️ Tidak ada dokumen untuk diindeks.")
        return

    print(f"📦 Mengindeks {len(all_documents)} chunks ke ChromaDB (ini mungkin memakan waktu)...")
    
    # Ingest ke ChromaDB
    try:
        vectorstore = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=DB_DIR,
            collection_name=COLLECTION_NAME
        )
        print(f"✅ Selesai! Database disimpan di: {DB_DIR}")
    except Exception as e:
        print(f"❌ Gagal melakukan ingest ke ChromaDB: {e}")

if __name__ == "__main__":
    main()
