import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Directories
PROCESSED_DIR = os.path.abspath("../data/processed")
RAW_DIR = os.path.abspath("../data/raw_pdfs")
#CSV_DATASET_FILE = os.path.join(PROCESSED_DIR, "dreams_interpretations.csv")  # Updated input file
CSV_DATASET_FILE = os.path.join(RAW_DIR, "dreams_interpretations.csv")
TOKENIZED_OUTPUT_DIR = os.path.join(PROCESSED_DIR, "tokenized_dataset")

# Ensure the tokenized dataset directory exists
os.makedirs(TOKENIZED_OUTPUT_DIR, exist_ok=True)

# Load tokenizer
MODEL_NAME = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure a padding token is set (GPT-2 does not have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    """Tokenizes text data."""
    return tokenizer(
        examples["text"],
        padding="longest",
        truncation=True,
        max_length=100
    )

def tokenize_dataset():
    """Tokenizes dataset from CSV for fine-tuning."""
    if not os.path.exists(CSV_DATASET_FILE):
        raise FileNotFoundError(f"Dataset file not found: {CSV_DATASET_FILE}")

    # Load dataset from CSV
    df = pd.read_csv(CSV_DATASET_FILE)

    # Ensure columns exist
    if "Dream Symbol" not in df.columns or "Interpretation" not in df.columns:
        raise ValueError("CSV must contain 'Dream Symbol' and 'Interpretation' columns.")

    # Merge Dream and Interpretation into a single text prompt
    df["text"] = "Dream: " + df["Dream Symbol"] + "\nInterpretation: " + df["Interpretation"] + "\n"

    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_pandas(df[["text"]])

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Print sample for debugging
    tokens = tokenizer(df["text"].iloc[0])
    print("Tokenized sample:", tokens)

    # Save tokenized dataset
    tokenized_dataset.save_to_disk(TOKENIZED_OUTPUT_DIR)
    print(f"Tokenized dataset saved to {TOKENIZED_OUTPUT_DIR}")

if __name__ == "__main__":
    tokenize_dataset()
