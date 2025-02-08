import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import torch
from datasets import Dataset, load_dataset

# Directories
PROCESSED_DIR = Path("../data/processed")
DATASET_FILE = PROCESSED_DIR / "dataset.jsonl"
TOKENIZED_OUTPUT_FILE = PROCESSED_DIR / "tokenized_dataset"

# Load LLaMA 2 tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Adjust based on the model you're using
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    """Tokenizes text data for LLaMA 2."""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def tokenize_dataset():
    """Tokenizes dataset for fine-tuning LLaMA 2."""
    if not DATASET_FILE.exists():
        print(f"Dataset file {DATASET_FILE} not found. Run dataset preparation first.")
        return

    # Load dataset
    dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Save tokenized dataset in Hugging Face format
    tokenized_dataset.save_to_disk(str(TOKENIZED_OUTPUT_FILE))
    print(f"Tokenized dataset saved to {TOKENIZED_OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize dataset for LLaMA 2 fine-tuning.")
    args = parser.parse_args()

    tokenize_dataset()
