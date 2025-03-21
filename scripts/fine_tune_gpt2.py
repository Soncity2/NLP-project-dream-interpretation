import yaml
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
CONFIG_PATH = Path("../config/gpt2_training_config.yaml")
DATASET_DIR = Path("../data/raw_pdfs")
PROCESSED_DIR = Path("../data/processed")
#DATASET_FILE = PROCESSED_DIR / "dreams_interpretations.csv"
DATASET_FILE = DATASET_DIR / "dreams_interpretations.csv"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")

def load_training_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ["Dream", "Interpretation"]):
        raise ValueError("CSV must contain 'Dream' and 'Interpretation' columns.")
    df["text"] = df.apply(lambda row: f"Dream: {row['Dream']}\nInterpretation: {row['Interpretation']}", axis=1)
    dataset = Dataset.from_pandas(df[["text"]])
    return dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding="max_length", return_tensors="pt"),
        batched=True
    )

def fine_tune():
    config = load_training_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dataset = load_dataset(DATASET_FILE)
    split_dataset = dataset.train_test_split(test_size=0.1)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
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
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    logging.info(f"Fine-tuned model saved at {config['output_dir']}")

if __name__ == "__main__":
    fine_tune()