from datasets import Dataset
from transformers import AutoTokenizer
import os

# Directories
PROCESSED_DIR = os.path.abspath("../data/processed")
DATASET_FILE = os.path.join(PROCESSED_DIR, "dataset.txt")
TOKENIZED_OUTPUT_DIR = os.path.join(PROCESSED_DIR, "tokenized_dataset")

# Ensure the tokenized dataset directory exists
os.makedirs(TOKENIZED_OUTPUT_DIR, exist_ok=True)

# Load tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# LLaMA does not have a default pad token, so we set it manually
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    """Tokenizes text data for LLaMA 2."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

def tokenize_dataset():
    """Tokenizes dataset for fine-tuning LLaMA 2."""
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

    # Read dataset manually
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Convert text into a Hugging Face Dataset format
    dataset = Dataset.from_dict({"text": [line.strip() for line in lines if line.strip()]})

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Ensure the directory exists before saving
    os.makedirs(TOKENIZED_OUTPUT_DIR, exist_ok=True)

    # Save tokenized dataset
    tokenized_dataset.save_to_disk(TOKENIZED_OUTPUT_DIR)
    print(f"Tokenized dataset saved to {TOKENIZED_OUTPUT_DIR}")

if __name__ == "__main__":
    tokenize_dataset()
