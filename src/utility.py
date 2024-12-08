import pandas as pd

# Unified function to load different file types (CSV, TSV, JSON)
def load_data(file_path):
    # Get the file extension to determine how to load the file
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'csv':
        data = pd.read_csv(file_path)  # Load CSV file
    elif file_extension == 'tsv':
        data = pd.read_csv(file_path, sep='\t')  # Load TSV file with tab delimiter
    elif file_extension == 'json':
        data = pd.read_json(file_path, encoding="utf-8")  # Load JSON file
    else:
        raise ValueError("Unsupported file type. Please provide a CSV, TSV, or JSON file.")
    
    return data