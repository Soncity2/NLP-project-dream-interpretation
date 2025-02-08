import os
import json
import argparse
from pathlib import Path
from PyPDF2 import PdfReader

# Directories
RAW_PDF_DIR = Path("../data/raw_pdfs")
PROCESSED_DIR = Path("../data/processed")

# Ensure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def process_pdfs():
    """Extract text from all PDFs in the raw_pdfs directory and save them."""
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            output_file = PROCESSED_DIR / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"filename": pdf_file.name, "text": text}, f, ensure_ascii=False, indent=4)
            print(f"Processed: {pdf_file.name} -> {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDFs and store them in structured format.")
    parser.add_argument("--dir", type=str, default=str(RAW_PDF_DIR), help="Directory containing raw PDFs.")
    args = parser.parse_args()

    RAW_PDF_DIR = Path(args.dir)
    process_pdfs()
