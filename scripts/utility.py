import re
import csv

def format_model_name(name):
    # Convert to lowercase
    name = name.lower()
    # Remove non-alphanumeric characters except numbers
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def write_dream_analysis_to_csv(data, metrics_filename='model_metrics.csv',
                                samples_filename='interpretation_results.csv'):
    """
    Writes model metrics and dream interpretation samples from a given JSON-like dictionary to CSV files.

    Parameters:
        data (dict): The JSON data containing model results and samples.
        metrics_filename (str): Filename for saving overall model metrics.
        samples_filename (str): Filename for saving individual dream interpretations.
    """

    # Save model-level metrics
    with open(metrics_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "SacreBLEU", "ROUGE-L", "BERTScore-F1", "Perplexity"])
        writer.writerow([
            data.get("model", ""),
            data.get("SacreBLEU", ""),
            data.get("ROUGE-L", ""),
            data.get("BERTScore-F1", ""),
            data.get("Perplexity", "")
        ])

    # Save dream interpretation samples
    with open(samples_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Dream", "Generated Interpretation"])
        for sample in data.get("samples", []):
            writer.writerow([
                sample.get("dream", ""),
                sample.get("generated", "")
            ])

