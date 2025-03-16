import yaml
import logging
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from pathlib import Path


logging.basicConfig(level=logging.INFO)
PROCESSED_DIR = Path("../data/processed")
DATASET_FILE = PROCESSED_DIR / "dreams_interpretations.csv"
CONFIG_PATH = Path("../config/t5_training_config.yaml")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def load_training_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ["Dream", "Interpretation"]):
        raise ValueError("CSV must contain 'Dream' and 'Interpretation' columns.")
    df["input_text"] = "interpret dream: " + df["Dream"]
    df["target_text"] = df["Interpretation"]
    dataset = Dataset.from_pandas(df[["input_text", "target_text"]])
    return dataset.map(
        lambda x: {
            "input_ids": tokenizer(x["input_text"], truncation=True, max_length=512, padding="max_length")["input_ids"],
            "labels": tokenizer(x["target_text"], truncation=True, max_length=512, padding="max_length")["input_ids"]
        },
        batched=True
    )

def fine_tune_t5():
    config = load_training_config()
    dataset = load_dataset(DATASET_FILE)
    split_dataset = dataset.train_test_split(test_size=0.1)
    training_args = TrainingArguments(
        output_dir="../models/fine_tuned_t5",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        evaluation_strategy="epoch",
        fp16=torch.cuda.is_available(),
        logging_dir="../logs/t5"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer
    )
    trainer.train()
    model.save_pretrained("../models/fine_tuned_t5")
    tokenizer.save_pretrained("../models/fine_tuned_t5")
    logging.info("T5 model saved.")

if __name__ == "__main__":
    fine_tune_t5()