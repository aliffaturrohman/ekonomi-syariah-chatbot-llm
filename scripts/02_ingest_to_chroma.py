import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

INPUT_MD_DIR = "../data/processed/markdown_converted_from_pdf" 
DB_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

def ingest_to_database(INPUT_MD_PATH, DB_DIR, COLLECTION_NAME):
    if not os.path.exists(INPUT_MD_PATH):
        print(f"File tidak ditemukan: {INPUT_MD_PATH}")
        return

    with open(INPUT_MD_PATH, "r", encoding="utf-8") as f:
        clean_text = f.read()
        
    print(f"⚙️  Memuat Model Embedding: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Chunking text")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=[
            "\n\n",
            "\n#",
            "\n1. ",
            "\n- ",
            "\n* ",
            "\n   a. ",
            ". ",
            " ",
            ""
        ],
        keep_separator=True
    )

    chunks = text_splitter.split_text(clean_text)
    docs = [Document(page_content=c, metadata={"source": INPUT_MD_PATH}) for c in chunks]
    
    print(f"   Terbentuk {len(docs)} potongan (chunks).")
    print(f"💾 Menyimpan Vector ke ChromaDB di folder: {DB_DIR}")
    
    if os.path.exists(DB_DIR):
        print("⚠️ Database lama terdeteksi. Data baru akan ditambahkan (append).")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print("✅ Selesai! Data buku sudah masuk ke Vector Database.")
    print("   Siap untuk tahap selanjutnya: Generate RAFT Dataset.")

def main():
    for filename in os.listdir(INPUT_MD_DIR):
        if filename.endswith(".md"):
            input_md_path = os.path.join(INPUT_MD_DIR, filename)
            ingest_to_database(input_md_path, DB_DIR, COLLECTION_NAME)

if __name__ == "__main__":
    ingest_to_database("../data/processed/02_cleaning_markdown_text_heading/Ekonomi Syariah E-Book [UINSA]_Bab_1_[12-26].md", DB_DIR, COLLECTION_NAME)