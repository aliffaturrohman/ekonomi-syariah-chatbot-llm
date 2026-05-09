import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

# --- KONFIGURASI ---
INPUT_MD_DIR = "../data/processed/markdown_converted_from_pdf"

# Lokasi Database
DB_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

def reset_database():
    """Hapus database lama agar bersih."""
    if os.path.exists(DB_DIR):
        print(f"🗑️  MENGHAPUS Database lama di: {DB_DIR} ...")
        try:
            shutil.rmtree(DB_DIR)
            print("   ✅ Database lama berhasil dihapus.")
        except Exception as e:
            print(f"   ⚠️ Gagal menghapus database: {e}")
    else:
        print("   ℹ️ Database belum ada, akan dibuat baru.")

def ingest_to_database(file_path, embedding_model, vectorstore_exists=False):
    """Memproses satu file MD dan memasukkannya ke DB."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n#", "\n1. ", "\n- ", ". ", " ", ""],
        keep_separator=True
    )

    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"source": os.path.basename(file_path)}) for c in chunks]

    if not vectorstore_exists:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=DB_DIR,
            collection_name=COLLECTION_NAME
        )
        return len(docs), True
    else:
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model
        )
        vectorstore.add_documents(docs)
        return len(docs), True

def main():
    reset_database()

    print(f"⚙️  Memuat Model Embedding: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if not os.path.exists(INPUT_MD_DIR):
        print(f"❌ Folder tidak ditemukan: {INPUT_MD_DIR}")
        return

    all_files = [os.path.join(INPUT_MD_DIR, f) for f in os.listdir(INPUT_MD_DIR) if f.endswith(".md")]

    if not all_files:
        print("❌ Tidak ada file .md ditemukan.")
        return

    print(f"🚀 Memulai Ingest untuk {len(all_files)} file (Full Raw Markdown)...")
    
    total_chunks = 0
    vs_exists = False
    for file_path in tqdm(all_files, desc="Processing Files"):
        num_chunks, vs_exists = ingest_to_database(file_path, embeddings, vs_exists)
        total_chunks += num_chunks
    
    print("\n" + "="*40)
    print(f"✅ SELESAI! Total {total_chunks} potongan teks masuk ke Database.")
    print(f"📂 Lokasi DB: {DB_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()