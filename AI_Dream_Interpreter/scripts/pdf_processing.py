import os
import argparse
from pathlib import Path
from PyPDF2 import PdfReader

# Directories
RAW_PDF_DIR = Path("../data/raw_pdfs")
PROCESSED_DIR = Path("../data/processed")

print(PROCESSED_DIR)

# Ensure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()  # Remove leading/trailing spaces
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def process_pdfs():
    """Extracts text from all PDFs and saves them as .txt files."""
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            output_file = PROCESSED_DIR / f"{pdf_file.stem}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text + "\n")  # Ensure newline at the end
            print(f"Processed: {pdf_file.name} -> {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDFs and store them in plaintext format.")
    parser.add_argument("--dir", type=str, default=str(RAW_PDF_DIR), help="Directory containing raw PDFs.")
    args = parser.parse_args()

    RAW_PDF_DIR = Path(args.dir)
    process_pdfs()
