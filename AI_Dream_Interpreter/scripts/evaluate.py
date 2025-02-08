import torch
import argparse
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric

# Directories
MODEL_DIR = Path("../models/fine_tuned_llama2")
PROCESSED_DIR = Path("../data/processed")
EVAL_RESULTS_FILE = PROCESSED_DIR / "evaluation_results.json"

# Model name (adjust if necessary)
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# Load BLEU and ROUGE metrics
bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge")

def generate_text(model, tokenizer, prompt, max_length=200):
    """Generates text using the fine-tuned model."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate():
    """Evaluates the fine-tuned model."""
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

    # Define test prompts for evaluation
    test_prompts = [
        "Explain the impact of cloud computing on modern businesses.",
        "What are the key benefits of AI in healthcare?",
        "Describe the history of artificial intelligence.",
    ]

    # Generate responses
    results = []
    for prompt in test_prompts:
        generated_text = generate_text(model, tokenizer, prompt)
        results.append({"prompt": prompt, "generated_text": generated_text})
        print(f"Prompt: {prompt}\nGenerated: {generated_text}\n")

    # Compute evaluation metrics (dummy ground truth for example purposes)
    references = [["Cloud computing has transformed modern businesses..."],
                  ["AI in healthcare offers improved diagnostics..."],
                  ["Artificial intelligence has a long history dating back to..."]]

    generated_texts = [r["generated_text"] for r in results]

    bleu_score = bleu_metric.compute(predictions=generated_texts, references=references)
    rouge_score = rouge_metric.compute(predictions=generated_texts, references=references)

    # Save evaluation results
    eval_results = {
        "BLEU": bleu_score,
        "ROUGE": rouge_score,
        "generated_samples": results
    }

    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Evaluation results saved to {EVAL_RESULTS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned LLaMA 2 model.")
    args = parser.parse_args()

    evaluate()
