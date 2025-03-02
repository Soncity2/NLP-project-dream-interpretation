import yaml
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from pathlib import Path

# Paths
CONFIG_PATH = Path("../config/gpt2_training_config.yaml")
PROCESSED_DIR = Path("../data/processed")
RAW_DIR = Path("../data/raw_pdfs")
#CSV_DATASET_FILE = os.path.join(PROCESSED_DIR, "dreams_interpretations.csv")  # Updated input file
DATASET_FILE = RAW_DIR / "dreams_interpretations.csv"

def load_training_config():
    """Loads training configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Fix padding issue
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Function to load and tokenize dataset from CSV
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)  # ✅ Load CSV
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    if "Dream Symbol" not in df.columns or "Interpretation" not in df.columns:
        raise ValueError("CSV must contain 'Dream Symbol' and 'Interpretation' columns.")

    # ✅ Merge columns into a "text" field formatted correctly
    df["text"] = df.apply(lambda row: f"Dream: {row['Dream Symbol']}. Interpretation: {row['Interpretation']}", axis=1)

    def tokenize_function(examples):
        encodings = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
        encodings["labels"] = encodings["input_ids"].copy()  # Labels should match input IDs for GPT-2 training
        return encodings

    dataset = Dataset.from_pandas(df[["text"]])  # Convert Pandas DataFrame to Hugging Face Dataset
    dataset = dataset.map(tokenize_function, batched=True)
    return dataset

def fine_tune():
    """Fine-tunes GPT-2 on GPU with CSV input."""

    # Load training config
    config = load_training_config()

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ensure pad_token is properly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Data collator for efficient training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal Language Model requires MLM=False
        return_tensors="pt"
    )

    # Load tokenized dataset
    dataset = load_dataset(DATASET_FILE)
    if dataset is None:
        print("Dataset loading failed. Exiting.")
        return

    # Split dataset into train (90%) and eval (10%)
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        learning_rate=float(config["learning_rate"]),
        max_grad_norm=1.0,
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_total_limit=config["save_total_limit"],
        evaluation_strategy=config["evaluation_strategy"],
        num_train_epochs=config["num_train_epochs"],
        fp16=torch.cuda.is_available(),  # Enable FP16 if GPU supports it
        bf16=False,  # Enable BF16 if GPU supports it
        logging_dir=config["log_dir"],
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train model
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Fine-tuned model saved at {config['output_dir']}")


if __name__ == "__main__":
    fine_tune()
