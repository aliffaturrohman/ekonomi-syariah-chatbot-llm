import sys
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- KONFIGURASI (DISAMAKAN DENGAN SCRIPT INGEST KAMU) ---
DB_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "ekonomi_syariah_dataset"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

def main():
    # 1. Cek folder database
    if not os.path.exists(DB_DIR):
        print(f"❌ Error: Folder database tidak ditemukan di: {os.path.abspath(DB_DIR)}")
        print("   Pastikan script ingest sudah dijalankan dan path-nya benar.")
        return

    print("⚙️  Memuat Model Embedding (Sama seperti saat Ingest)...")
    # Konfigurasi ini WAJIB sama persis dengan saat ingest agar vectornya cocok
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"📂 Membuka Database dari: {DB_DIR}")
    print(f"📚 Collection: {COLLECTION_NAME}")

    try:
        # Load Database
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
    except Exception as e:
        print(f"❌ Gagal load database: {e}")
        return

    print("\n✅ DATABASE TERHUBUNG!")
    print("   Silakan ketik pertanyaan untuk mengetes apakah data masuk.")
    print("   (Ketik 'exit' untuk berhenti)\n")

    # 3. Loop Pencarian
    while True:
        query = input("🔍 Cari (Query): ")
        if query.lower() in ['exit', 'quit', 'keluar']:
            break
        
        if not query.strip():
            continue

        print("   ⏳ Sedang mencari potongan yang relevan...")
        
        # Cari 3 hasil teratas
        results = vectorstore.similarity_search_with_score(query, k=3)

        if not results:
            print("   ⚠️ Tidak ditemukan hasil.")
        else:
            print(f"\n   Ditemukan {len(results)} hasil:")
            for i, (doc, score) in enumerate(results):
                # Score (L2 Distance): Makin kecil (mendekati 0) makin bagus/mirip.
                quality = "⭐⭐⭐" if score < 0.4 else "⭐⭐" if score < 0.7 else "⭐"
                
                print(f"\n   --- [Hasil #{i+1}] (Score: {score:.4f} {quality}) ---")
                print(f"   📄 Isi: {doc.page_content[:3000].replace(chr(10), ' ')}...") # Preview 300 huruf
                print(f"   📂 Sumber: {doc.metadata.get('source', 'Unknown')}")
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()