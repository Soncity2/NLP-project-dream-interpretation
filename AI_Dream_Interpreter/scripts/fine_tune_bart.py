import yaml
import torch
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
CONFIG_PATH = Path("../config/bart_training_config.yaml")  # create a new config if desired
PROCESSED_DIR = Path("../data/processed")
DATASET_FILE = PROCESSED_DIR / "dreams_interpretations.csv"

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

def load_training_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ["Dream", "Interpretation"]):
        raise ValueError("CSV must contain 'Dream' and 'Interpretation' columns.")
    # Construct a prompt: you can experiment with different formulations.
    df["text"] = df.apply(lambda row: f"Dream: {row['Dream']}\nInterpretation: {row['Interpretation']}", axis=1)
    dataset = Dataset.from_pandas(df[["text"]])
    # Tokenize the dataset for an encoder-decoder model
    def tokenize_fn(example):
        model_inputs = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
        # For a generation task, labels can simply be the tokenized version of the target text,
        # which in this simple case is the full text. You might choose to separate input and target.
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    dataset = dataset.map(tokenize_fn, batched=True)
    return dataset

def fine_tune_bart():
    config = load_training_config()
    dataset = load_dataset(DATASET_FILE)
    split_dataset = dataset.train_test_split(test_size=0.1)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
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
        fp16=torch.cuda.is_available(),
        logging_dir=config["log_dir"]
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
    fine_tune_bart()
