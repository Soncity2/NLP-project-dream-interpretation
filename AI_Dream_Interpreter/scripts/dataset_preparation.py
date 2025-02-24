import re
import json
import nltk
from pathlib import Path

nltk.download("punkt_tab")

# Directories
PROCESSED_DIR = Path("../data/processed")
PDF_TXT_PATH = PROCESSED_DIR / "dreams.txt"
OUTPUT_FILE = PROCESSED_DIR / "dataset.jsonl"


def extract_dreams_interpretations(text):
    """Extracts dream descriptions and interpretations from Freud’s raw text."""
    sentences = nltk.sent_tokenize(text)

    dream_patterns = [r"dream", r"vision", r"nightmare", r"asleep", r"dreamed"]
    interpretation_patterns = [r"interpreted", r"symbolizes", r"means", r"indicates", r"represents"]

    dream_interpretation_pairs = []
    current_dream = None
    current_interpretation = None

    for i, sentence in enumerate(sentences):
        # Identify a dream-related sentence
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in dream_patterns):
            current_dream = sentence.strip()

        # Identify an interpretation-related sentence following a dream
        if current_dream and any(re.search(pattern, sentence, re.IGNORECASE) for pattern in interpretation_patterns):
            current_interpretation = sentence.strip()
            dream_interpretation_pairs.append({"prompt": current_dream, "response": current_interpretation})
            current_dream = None  # Reset for next pair

    return dream_interpretation_pairs


def save_to_jsonl():
    """Processes the Freud text and saves extracted pairs into JSONL."""
    with open(PDF_TXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    pairs = extract_dreams_interpretations(text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in pairs:
            json.dump(pair, f)
            f.write("\n")

    print(f"Extracted {len(pairs)} dream-interpretation pairs. Saved at {OUTPUT_FILE}")


if __name__ == "__main__":
    save_to_jsonl()