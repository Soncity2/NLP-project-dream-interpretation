import yaml
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk

# Paths
CONFIG_PATH = Path("config/training_config.yaml")
TOKENIZED_DATASET_DIR = Path("data/processed/tokenized_dataset")


def load_training_config():
    """Loads training configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def fine_tune():
    """Fine-tunes LLaMA 2 on the tokenized dataset."""

    # Load training config
    config = load_training_config()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load tokenized dataset
    dataset = load_from_disk(str(TOKENIZED_DATASET_DIR))

    # Training arguments from config file
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        evaluation_strategy=config["evaluation_strategy"],
        eval_steps=config["eval_steps"],
        warmup_steps=config["warmup_steps"],
        num_train_epochs=config["num_train_epochs"],
        fp16=config["fp16"],
        logging_dir=config["log_dir"],
        report_to=config["report_to"]
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # Train model
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Fine-tuned model saved at {config['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 2 on processed dataset.")
    args = parser.parse_args()

    fine_tune()
