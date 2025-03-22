import yaml
import logging
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from pathlib import Path


logging.basicConfig(level=logging.INFO)
DATASET_DIR = Path("../data/raw_pdfs")
PROCESSED_DIR = Path("../data/processed")
#DATASET_FILE = PROCESSED_DIR / "dreams_interpretations.csv"
DATASET_FILE = DATASET_DIR / "dreams_interpretations.csv"
CONFIG_PATH = Path("../config/t5_training_config.yaml")

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

def load_training_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ["Dream", "Interpretation"]):
        raise ValueError("CSV must contain 'Dream' and 'Interpretation' columns.")

    df["input"] = df["Dream"].apply(lambda x: f"interpret dream: {x}")
    df["target"] = df["Interpretation"]

    dataset = Dataset.from_pandas(df[["input", "target"]])

    def tokenize(example):
        model_inputs = tokenizer(example["input"], max_length=512, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example["target"], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(tokenize, batched=True)

def fine_tune_t5():
    config = load_training_config()
    dataset = load_dataset(DATASET_FILE)
    split_dataset = dataset.train_test_split(test_size=0.1)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        logging_steps=config["logging_steps"],
        evaluation_strategy=config["evaluation_strategy"],
        save_total_limit=config["save_total_limit"],
        max_grad_norm=config["max_grad_norm"],
        fp16=torch.cuda.is_available(),
        logging_dir=config["log_dir"],
        report_to="tensorboard"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained("../models/fine_tuned_t5")
    tokenizer.save_pretrained("../models/fine_tuned_t5")
    logging.info("T5 model saved.")

if __name__ == "__main__":
    fine_tune_t5()