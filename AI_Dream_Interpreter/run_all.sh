#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=================================="
echo "Starting fine-tuning for GPT-2..."
echo "=================================="
python scripts/fine_tune_gpt2.py
if [ $? -ne 0 ]; then
    echo "Error during GPT-2 fine-tuning. Exiting."
    exit 1
fi

echo "=================================="
echo "Starting fine-tuning for BART..."
echo "=================================="
python scripts/fine_tune_bart.py
if [ $? -ne 0 ]; then
    echo "Error during BART fine-tuning. Exiting."
    exit 1
fi

echo "=================================="
echo "Starting model evaluation..."
echo "=================================="
python scripts/evaluate_models.py
if [ $? -ne 0 ]; then
    echo "Error during evaluation. Exiting."
    exit 1
fi

echo "=================================="
echo "All processes completed successfully."
echo "=================================="
