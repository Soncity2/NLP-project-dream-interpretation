import os
import argparse
from pathlib import Path

# Directories
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "dataset.txt"


def clean_text(text):
    """Cleans text by removing extra spaces and ensuring proper formatting."""
    text = text.strip()  # Remove leading/trailing spaces
    text = " ".join(text.split())  # Normalize spaces
    return text


def split_text(text, chunk_size=512):
    """Splits text into smaller chunks for better processing."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))  # Create chunks of words
    return chunks


def prepare_dataset():
    """Reads processed .txt files and creates a structured dataset."""
    txt_files = list(PROCESSED_DIR.glob("*.txt"))

    if not txt_files:
        print("No processed text files found.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
                cleaned_text = clean_text(text)  # Apply text cleaning
                text_chunks = split_text(cleaned_text)  # Break into smaller parts

                for chunk in text_chunks:
                    output_file.write(chunk + "\n\n")  # Separate chunks with newlines

    print(f"Dataset saved to {OUTPUT_FILE} with {len(text_chunks)} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset from extracted text files.")
    args = parser.parse_args()

    prepare_dataset()
