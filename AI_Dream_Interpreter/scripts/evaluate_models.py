import torch
import argparse
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate  # Hugging Face evaluation library

# Directories
MODEL_DIR = Path("../models/fine_tuned_gpt2")
PROCESSED_DIR = Path("../data/processed")
EVAL_RESULTS_FILE = PROCESSED_DIR / "results/evaluation_results.json"
DREAMS_FILE = Path("../data/dreams_list/Freud_Dreams_Interpretations.txt")

# Model name (adjust if necessary)
MODEL_NAME = "openai-community/gpt2"

# Load evaluation metrics using `evaluate` library
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
perplexity_metric = evaluate.load("perplexity")
bertscore_metric = evaluate.load("bertscore")

def read_dreams_file(dreams_file):
    """Reads the dream interpretation file and extracts pairs of dreams and interpretations."""
    dreams = []
    interpretations = []
    with open(dreams_file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                dream, interpretation = line.strip().split(":", 1)
                dreams.append(dream.strip())
                interpretations.append(interpretation.strip())
    return dreams, interpretations

def generate_text(model, tokenizer, prompt, max_length=100):
    """Generates text using the fine-tuned model."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_models():
    """Evaluates the fine-tuned model."""
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

    # Load dreams and interpretations
    dreams, true_interpretations = read_dreams_file(DREAMS_FILE)

    # Generate responses
    generated_interpretations = []
    results = []
    for dream in dreams:
        generated_text = generate_text(model, tokenizer, dream)
        generated_interpretations.append(generated_text)
        results.append({"dream": dream, "generated_interpretation": generated_text})
        print(f"Dream: {dream}\nGenerated Interpretation: {generated_text}\n")

    # Compute evaluation metrics using Hugging Face `evaluate`
    bleu_score = bleu_metric.compute(predictions=generated_interpretations, references=true_interpretations)
    rouge_score = rouge_metric.compute(predictions=generated_interpretations, references=true_interpretations)
    perplexity_score = perplexity_metric.compute(predictions=generated_interpretations, model_id=MODEL_NAME)
    bertscore_score = bertscore_metric.compute(
        predictions=generated_interpretations,
        references=true_interpretations,
        lang="en"
    )

    # Save evaluation results
    eval_results = {
        "BLEU": bleu_score,
        "ROUGE": rouge_score,
        "Perplexity": perplexity_score,
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
