import torch
import argparse
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate  # Hugging Face evaluation library

# Directories
MODEL_DIR = Path("../models/fine_tuned_gpt2")
PROCESSED_DIR = Path("../data/processed")
EVAL_RESULTS_FILE = PROCESSED_DIR / "results/evaluation_results.json"
DREAMS_FILE = Path("../data/dreams_list/Freud_Dreams_Interpretations.txt")

# Load evaluation metrics using `evaluate` library
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")


def read_dreams_file(dreams_file):
    """
    Reads the dream interpretation file and extracts pairs of dreams and interpretations.
    Assumes format: "<dream>: <interpretation>" on each line.
    """
    dreams = []
    interpretations = []

    with open(dreams_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        # Ensure valid format: "Dream: Interpretation"
        if ":" in line:
            dream, interpretation = line.split(":", 1)  # Split only on first colon
            dreams.append(dream.strip())  # Remove extra spaces
            interpretations.append(interpretation.strip())

    return dreams, interpretations



def generate_text(model, tokenizer, prompt, max_length=100):
    """
    Generates text while ensuring stable behavior.
    Handles CUDA errors, corrects attention masking, and prevents infinite punctuation spam.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # Ensure model is on the correct device

    # Ensure tokenizer has correct padding tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        # Tokenize input safely
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Ensure correct dtype & device


        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Fix model echoing issues (if it repeats prompt exactly, regenerate)
        if generated_text.strip().lower() == prompt.strip().lower():
            print("Warning: Model is echoing the prompt. Retrying with adjusted temperature...")
            return generate_text(model, tokenizer, prompt, max_length)

        return generated_text

    except torch.cuda.CudaError as e:
        print(f"CUDA Error: {e} - Falling back to CPU.")
        model.to("cpu")  # Move model to CPU if CUDA crashes
        return generate_text(model, tokenizer, prompt, max_length)

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return "Error: An unexpected issue occurred during text generation."


def evaluate_models():
    """
    Evaluates the fine-tuned model and computes BLEU, ROUGE, and BERTScore metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)

    # Load dreams and interpretations
    dreams, true_interpretations = read_dreams_file(DREAMS_FILE)

    # Ensure we have data
    if not dreams or not true_interpretations:
        print("Error: No dreams or interpretations found. Check the input file format.")
        return

    # Generate responses
    generated_interpretations = []
    results = []

    for dream in dreams:
        generated_text = generate_text(model, tokenizer, dream)
        generated_interpretations.append(generated_text)
        results.append({"dream": dream, "generated_interpretation": generated_text})
        print(f"Dream: {dream}\nGenerated Interpretation: {generated_text}\n")

    # Compute evaluation metrics using Hugging Face `evaluate`
    try:
        bleu_score = bleu_metric.compute(predictions=generated_interpretations, references=true_interpretations)
        rouge_score = rouge_metric.compute(predictions=generated_interpretations, references=true_interpretations)
        bertscore_score = bertscore_metric.compute(
            predictions=generated_interpretations,
            references=true_interpretations,
            lang="en"
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    # Save evaluation results
    eval_results = {
        "BLEU": bleu_score,
        "ROUGE": rouge_score,
        "BERTScore": bertscore_score,
        "generated_samples": results
    }

    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved to {EVAL_RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned LLM model.")
    args = parser.parse_args()
    evaluate_models()
