import csv
import os

# Directories
PROCESSED_DIR = os.path.abspath("../data/processed")
DATASET_FILE = os.path.join(PROCESSED_DIR, "dreams_freudian_structured.txt")
CSV_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "dreams_interpretations.csv")

# Ensure the processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def save_to_csv():

        """Converts structured TXT file into a CSV with 'Dream' and 'Interpretation' columns."""
        if not os.path.exists(DATASET_FILE):
            raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

        data = []

        # Read and process the TXT file
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    dream, interpretation = line.strip().split(":", 1)
                    data.append([dream.strip(), interpretation.strip()])

        # Save to CSV
        with open(CSV_OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dream", "Interpretation"])  # Column headers
            writer.writerows(data)

        print(f"CSV file saved to: {CSV_OUTPUT_FILE}")

if __name__ == "__main__":
    save_to_csv()