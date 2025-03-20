import csv
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
PROCESSED_DIR = os.path.abspath("../data/processed")
DATASET_FILE = os.path.join(PROCESSED_DIR, "dreams_freudian_structured.txt")
CSV_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "dreams_interpretations.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_text(text):
    return " ".join(text.split()).strip()

def save_to_csv():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")
    data = []
    skipped_lines = 0
    total_lines = 0
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line or ":" not in line:
                skipped_lines += 1
                continue
            try:
                dream, interpretation = line.split(":", 1)
                dream = clean_text(dream)
                interpretation = clean_text(interpretation)
                if len(dream) < 5 or len(interpretation) < 5:
                    skipped_lines += 1
                    continue
                data.append([dream, interpretation])
            except Exception as e:
                logging.warning(f"Error processing line '{line}': {e}")
                skipped_lines += 1
    with open(CSV_OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dream", "Interpretation"])
        writer.writerows(data)
    logging.info(f"Processed {len(data)} valid entries out of {total_lines} lines.")
    logging.info(f"Skipped {skipped_lines} lines due to formatting or length issues.")
    logging.info(f"CSV file saved to: {CSV_OUTPUT_FILE}")

if __name__ == "__main__":
    save_to_csv()