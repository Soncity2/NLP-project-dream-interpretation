import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, T5ForConditionalGeneration, AutoTokenizer
import evaluate
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
MODEL_DIR = Path("../models/fine_tuned_gpt2")
PROCESSED_DIR = Path("../data/processed")
EVAL_RESULTS_FILE = PROCESSED_DIR / "results/evaluation_results.json"
DREAMS_FILE = PROCESSED_DIR / "dreams_freudian_structured.txt"

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def read_dreams_file(dreams_file):
    dreams, interpretations = [], []
    with open(dreams_file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                dream, interp = line.split(":", 1)
                dreams.append(dream.strip())
                interpretations.append(interp.strip())
    logging.info(f"Loaded {len(dreams)} dream-interpretation pairs.")
    return dreams, interpretations

def generate_text(model, tokenizer, prompt, max_length=512, is_t5=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = f"interpret dream: {prompt}" if is_t5 else f"Dream: {prompt}\nInterpretation:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()

def evaluate_models(model_dir, model_name, is_t5=False):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = (T5ForConditionalGeneration if is_t5 else AutoModelForCausalLM).from_pretrained(model_dir)
    dreams, true_interps = read_dreams_file(DREAMS_FILE)
    generated_interps = [generate_text(model, tokenizer, dream, is_t5=is_t5) for dream in dreams[:10]]
    bleu = bleu_metric.compute(predictions=generated_interps, references=true_interps[:10])
    rouge = rouge_metric.compute(predictions=generated_interps, references=true_interps[:10])
    bertscore = bertscore_metric.compute(predictions=generated_interps, references=true_interps[:10], lang="en")
    results = {
        "model": model_name,
        "BLEU": bleu["bleu"],
        "ROUGE-L": rouge["rougeL"],
        "BERTScore-F1": sum(bertscore["f1"]) / len(bertscore["f1"]),
        "samples": [{"dream": d, "generated": g} for d, g in zip(dreams[:10], generated_interps)]
    }
    EVAL_RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    return results

def compare_models():
    gpt2_res = evaluate_models(MODEL_DIR, "GPT-2")
    t5_res = evaluate_models(Path("../models/fine_tuned_t5"), "T5", is_t5=True)
    logging.info("Comparison Results:")
    for res in [gpt2_res, t5_res]:
        logging.info(f"Model: {res['model']}")
        logging.info(f"BLEU: {res['BLEU']:.4f}")
        logging.info(f"ROUGE-L: {res['ROUGE-L']:.4f}")
        logging.info(f"BERTScore-F1: {res['BERTScore-F1']:.4f}\n")
        for sample in res["samples"][:3]:
            logging.info(f"Dream: {sample['dream']}\nGenerated: {sample['generated']}\n")

if __name__ == "__main__":
    compare_models()