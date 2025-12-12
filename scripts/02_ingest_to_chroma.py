import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

# --- KONFIGURASI BARU ---
# Folder tempat file Markdown BARU berada
INPUT_MD_DIR = "../data/processed/02_cleaning_markdown_text_heading" 

# Lokasi Database
DB_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

def reset_database():
    """Fungsi untuk menghapus database lama agar bersih."""
    if os.path.exists(DB_DIR):
        print(f"🗑️  MENGHAPUS Database lama di: {DB_DIR} ...")
        try:
            shutil.rmtree(DB_DIR) # Hapus folder beserta isinya
            print("   ✅ Database lama berhasil dihapus.")
        except Exception as e:
            print(f"   ⚠️ Gagal menghapus database: {e}")
    else:
        print("   ℹ️ Database belum ada, akan dibuat baru.")

def ingest_to_database(file_path, embedding_model):
    """Memproses satu file MD dan memasukkannya ke DB."""
    if not os.path.exists(file_path):
        return 0

    with open(file_path, "r", encoding="utf-8") as f:
        clean_text = f.read()

    # Chunking Config
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n#", "\n1. ", "\n- ", ". ", " ", ""],
        keep_separator=True
    )

    chunks = text_splitter.split_text(clean_text)
    
    # Masukkan metadata source agar kita tahu ini dari file mana
    docs = [Document(page_content=c, metadata={"source": file_path}) for c in chunks]

    # Simpan ke Chroma (Append mode, karena DB-nya sudah di-reset di awal main)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )
    
    return len(docs)

def main():
    # 1. Reset Database Dulu (Wajib biar data gak numpuk)
    reset_database()

    # 2. Cek Folder Input
    if not os.path.exists(INPUT_MD_DIR):
        print(f"❌ Folder input tidak ditemukan: {INPUT_MD_DIR}")
        return

    # 3. Load Model Embedding (Sekali saja biar cepet)
    print(f"⚙️  Memuat Model Embedding: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Loop Semua File Markdown di Folder Baru
    files = [f for f in os.listdir(INPUT_MD_DIR) if f.endswith(".md")]
    
    if not files:
        print(f"⚠️ Tidak ada file .md di {INPUT_MD_DIR}")
        return

    print(f"🚀 Memulai Ingest untuk {len(files)} file...")
    
    total_chunks = 0
    for filename in tqdm(files, desc="Processing Files"):
        file_path = os.path.join(INPUT_MD_DIR, filename)
        num_chunks = ingest_to_database(file_path, embeddings)
        total_chunks += num_chunks
    
    print("\n" + "="*40)
    print(f"✅ SELESAI! Total {total_chunks} potongan teks masuk ke Database.")
    print(f"📂 Lokasi DB: {DB_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()