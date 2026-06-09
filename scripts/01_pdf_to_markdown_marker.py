import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

PDF_DIR = "../data/raw/pdf"
OUTPUT_DIR = "../data/processed/markdown_converted_from_pdf"

def extract_pdf(PDF_PATH, OUTPUT_DIR):
    if not os.path.exists(PDF_PATH):
        print(f"Error: File tidak ditemukan di: {os.path.abspath(PDF_PATH)}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = os.path.basename(PDF_PATH).replace(".pdf", ".md")
    output_path = os.path.join(OUTPUT_DIR, file_name)

    # Periksa jika file sudah ada untuk menghindari proses ulang yang lama
    if os.path.exists(output_path):
        print(f"⏩ Skip: {file_name} sudah diproses.")
        return

    config_dict = {
        "output_format": "markdown",
        "languages": "id,ar",
        "disable_multiprocessing": False
    }
    
    print(f"🚀 Memproses: {file_name}")
    model_dict = create_model_dict()
    
    converter = PdfConverter(
        config=config_dict,
        artifact_dict=model_dict,
        processor_list=model_dict.get("processor_list"),
        renderer=model_dict.get("renderer")
    )

    try:
        rendered = converter(PDF_PATH)
        text, _, images = text_from_rendered(rendered)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"✅ Selesai: {file_name}")
        
    except Exception as e:
        print(f"❌ Error {file_name}: {e}")

def main():
    if not os.path.exists(PDF_DIR):
        print(f"❌ Folder PDF tidak ditemukan: {PDF_DIR}")
        return
        
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    print(f"📚 Ditemukan {len(pdf_files)} file PDF di {PDF_DIR}")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        extract_pdf(pdf_path, OUTPUT_DIR)

if __name__ == "__main__":
    main()
