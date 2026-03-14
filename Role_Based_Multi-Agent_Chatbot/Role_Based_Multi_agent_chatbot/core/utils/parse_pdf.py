import os
import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts and cleans text from a given PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"⚠️ Warning: Failed to extract text from page in {pdf_path}: {e}")
    except Exception as e:
        print(f"❌ Error opening PDF {pdf_path}: {e}")
    
    return text.strip()

def load_all_pdfs(folder_path: str) -> list:
    """Loads and returns text from all PDFs in the folder."""
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            raw_text = extract_text_from_pdf(full_path)
            texts.append({
                "filename": filename,
                "content": raw_text
            })
    return texts

def parse_single_pdf(file_path: str) -> list:
    """Parses a single PDF file and returns a list with one dict like load_all_pdfs()."""
    raw_text = extract_text_from_pdf(file_path)
    return [{
        "filename": os.path.basename(file_path),
        "content": raw_text
    }]
