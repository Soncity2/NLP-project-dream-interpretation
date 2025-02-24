import yaml
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk

# Paths
CONFIG_PATH = Path("../config/gpt2_training_config.yaml")
TOKENIZED_DATASET_DIR = Path("../data/processed/tokenized_dataset")


def load_training_config():
    """Loads training configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def fine_tune():
    """Fine-tunes LLaMA 2/GPT2 on GPU."""

    # Load training config
    config = load_training_config()

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Set a padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

    # Use a DataCollator to automatically shift labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # ✅ Set to False since we're using a causal language model (LLaMA 2)
    )

    # Load model with correct device mapping
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0"  # Ensures model is fully loaded on GPU
    )

    # Load tokenized dataset
    dataset = load_from_disk(str(TOKENIZED_DATASET_DIR))

    # Split dataset into train (90%) and eval (10%)
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=2,  # ✅ Increase for GPU
        gradient_accumulation_steps=4,
        learning_rate= float(config["learning_rate"]),
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        evaluation_strategy=config["evaluation_strategy"],
        eval_steps=config["eval_steps"],
        warmup_steps=config["warmup_steps"],
        num_train_epochs=config["num_train_epochs"],
        fp16=False,  # ✅ Enable FP16 for GPU
    #    bf16=False,  # ✅ Ensure BF16 is disabled
        logging_dir=config["log_dir"],
        report_to=config["report_to"]
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator  # Ensure loss is calculated correctly
    )

    # Train model
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Fine-tuned model saved at {config['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 2 on GPU.")
    args = parser.parse_args()

    fine_tune()
