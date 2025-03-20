import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
PROCESSED_DIR = os.path.abspath("../data/processed")
CSV_DATASET_FILE = os.path.join(PROCESSED_DIR, "dreams_interpretations.csv")
TOKENIZED_OUTPUT_DIR = os.path.join(PROCESSED_DIR, "tokenized_dataset")

os.makedirs(TOKENIZED_OUTPUT_DIR, exist_ok=True)
MODEL_NAME = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_dataset():
    if not os.path.exists(CSV_DATASET_FILE):
        raise FileNotFoundError(f"Dataset file not found: {CSV_DATASET_FILE}")
    df = pd.read_csv(CSV_DATASET_FILE)
    logging.info(f"Loaded CSV with {len(df)} rows.")
    if not all(col in df.columns for col in ["Dream", "Interpretation"]):
        raise ValueError("CSV must contain 'Dream' and 'Interpretation' columns.")
    df = df.dropna(subset=["Dream", "Interpretation"])
    df["text"] = "Dream: " + df["Dream"].str.strip() + "\nInterpretation: " + df["Interpretation"].str.strip()
    dataset = Dataset.from_pandas(df[["text"]])
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding="max_length"),
        batched=True,
        desc="Tokenizing dataset"
    )
    tokenized_dataset.save_to_disk(TOKENIZED_OUTPUT_DIR)
    logging.info(f"Tokenized dataset saved to {TOKENIZED_OUTPUT_DIR}")

if __name__ == "__main__":
    tokenize_dataset()