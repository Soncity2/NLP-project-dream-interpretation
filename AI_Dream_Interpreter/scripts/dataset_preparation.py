import os
import json
import argparse
from pathlib import Path

# Directories
PROCESSED_DIR = Path("../data/processed")
OUTPUT_FILE = PROCESSED_DIR / "dataset.jsonl"


def clean_text(text):
    """Cleans text by removing excessive spaces and unwanted characters."""
    return " ".join(text.split())


def prepare_dataset():
    """Reads processed JSON files and creates a structured dataset for fine-tuning."""
    dataset = []

    json_files = list(PROCESSED_DIR.glob("*.json"))
    if not json_files:
        print("No processed JSON files found.")
        return

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "text" in data:
                cleaned_text = clean_text(data["text"])
                dataset.append({"text": cleaned_text})

    # Save the dataset in JSONL format
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"Dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for LLaMA 2 fine-tuning.")
    args = parser.parse_args()

    prepare_dataset()
