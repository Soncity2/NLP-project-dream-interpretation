import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, BartForConditionalGeneration, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import evaluate
import logging
from utility import format_model_name, write_dream_analysis_to_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# Directories for the fine-tuned models
GPT2_MODEL_DIR = Path("../models/fine_tuned_gpt2")
BART_MODEL_DIR = Path("../models/fine_tuned_bart")
T5_MODEL_DIR = Path("../models/fine_tuned_t5")
PROCESSED_DIR = Path("../data/processed")
TEST_DREAMS_DIR = Path("../data/dreams_list")

DREAMS_FILE = TEST_DREAMS_DIR / "Freud_Dreams_Interpretations.txt"


# Load metrics: load BLEU with sacreBLEU configuration
bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")
perplexity_metric = evaluate.load("perplexity", module_type="metric")

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
    formatted_prompt = (
        f"interpret dream: {prompt}" if is_t5 else f"Dream: {prompt}\nInterpretation:"
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(device)
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
    return generated.strip()

def evaluate_models(model_dir, model_name, is_bart=False, is_t5=False):
    eval_results_file = Path(f"../results/evaluation_{model_name.lower()}.json")
    if is_t5:
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if is_bart:
            model = BartForConditionalGeneration.from_pretrained(model_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir)
    dreams, true_interps = read_dreams_file(DREAMS_FILE)
    generated_interps = [generate_text(model, tokenizer, dream) for dream in dreams[:10]]
    bleu = bleu_metric.compute(predictions=generated_interps, references=[[ref] for ref in true_interps[:10]])
    rouge = rouge_metric.compute(predictions=generated_interps, references=true_interps[:10])
    bertscore = bertscore_metric.compute(predictions=generated_interps, references=true_interps[:10], lang="en")
    if not (is_bart or is_t5):
        perplexity = perplexity_metric.compute(predictions=generated_interps, references=[true_interps[:10]], model_id=format_model_name(model_name))
    else:
        perplexity = {"mean_perplexity": None}
    results = {
        "model": model_name,
        "SacreBLEU": bleu["score"],
        "ROUGE-L": rouge["rougeL"],
        "BERTScore-F1": sum(bertscore["f1"]) / len(bertscore["f1"]),
        "Perplexity": perplexity["mean_perplexity"],
        "samples": [{"dream": d, "generated": g} for d, g in zip(dreams[:10], generated_interps)]
    }
    eval_results_file.parent.mkdir(exist_ok=True)
    with open(eval_results_file, "w") as f:
        json.dump(results, f, indent=4)

    # Load the JSON file from disk
    with open(eval_results_file, 'r', encoding='utf-8') as f:
        dream_json_data = json.load(f)

    write_dream_analysis_to_csv(dream_json_data,metrics_filename=Path(f"../results/model_metrics_{model_name}.csv"),
                                samples_filename=Path(f"../results/interpretation_results_{model_name}.csv"))
    return results

def compare_models():
    gpt2_res = evaluate_models(GPT2_MODEL_DIR, "GPT-2")
    t5_res = evaluate_models(T5_MODEL_DIR, "T5", is_t5=True)
    logging.info("Comparison Results:")
    for res in [gpt2_res, t5_res]:
        logging.info(f"Model: {res['model']}")
        logging.info(f"SacreBLEU: {res['SacreBLEU']:.4f}")
        logging.info(f"ROUGE-L: {res['ROUGE-L']:.4f}")
        logging.info(f"BERTScore-F1: {res['BERTScore-F1']:.4f}\n")
        logging.info(f"Perplexity: {res['Perplexity']:.4f}" if res["Perplexity"] is not None else "Perplexity: N/A")
        for sample in res["samples"][:3]:
            logging.info(f"Dream: {sample['dream']}\nGenerated: {sample['generated']}\n")

if __name__ == "__main__":
    compare_models()
