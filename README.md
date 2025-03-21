# **AI Dream Interpreter** 🌙💭✨  
**Fine-tune GPT‑2 and BART to interpret dreams based on structured text from PDFs.**

---

## 🌟 Overview  
This project fine-tunes two different language models – **GPT‑2** (a decoder‑only model) and **BART** (an encoder–decoder model) – using dream interpretation texts extracted from PDFs. Both models are trained on structured dream data, and you can later compare their interpretation outputs to explore how different architectures affect the results.

### Pipeline  
- ✅ Extract text from PDFs 📄  
- ✅ Preprocess & tokenize the dataset 🛠️  
- ✅ Fine-tune GPT‑2 and BART on dream-related data 🧠  
- ✅ Evaluate and compare model performance 📊

---

## 🚀 Setup Guide

### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/AvivSalomon/ai-dream-interpreter.git
cd ai-dream-interpreter
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```sh
python3 -m venv venv_dream
source venv_dream/bin/activate  # On Windows use: venv\Scripts\activate
```
### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```
```sh
pip install -e .
````
___
## 📂 Project Structure
```
ai-dream-interpreter/
│── config/                     
│   ├── gpt2_training_config.yaml       # Training hyperparameters for GPT-2
│   ├── bart_training_config.yaml       # Training hyperparameters for BART
│── data/                         
│   ├── raw_pdfs/                  # PDF files for fine-tuning
│   ├── processed/                 # Extracted & processed text & CSVs
│── models/                        # Trained models (e.g., fine_tuned_gpt2, fine_tuned_bart)
│── scripts/                      
│   ├── pdf_processing.py           # Extract text from PDFs
│   ├── dataset_preparation.py      # Prepare dataset from processed text
│   ├── tokenize_dataset.py         # Tokenize text dataset
│   ├── fine_tune.py                # Fine-tune GPT-2 on dream data
│   ├── fine_tune_bart.py           # Fine-tune BART on dream data
│   ├── evaluate_models.py          # Evaluate and compare models
│── logs/                          
│── requirements.txt               
│── setup.py                       
│── README.md                      
│── .gitignore
```
---
##  🎯 Usage Instructions

📄 1️⃣ Extract Text from PDFs
Place your PDF files in data/raw_pdfs/ and run:

```sh
python scripts/pdf_processing.py
Output: Extracted text stored in data/processed/ as .json files.
```


🛠 2️⃣ Prepare Dataset for Training
```sh
python scripts/dataset_preparation.py
Output: A structured dataset in data/processed/dataset.json.
```
🔢 3️⃣ Tokenize the Dataset
```sh
python scripts/tokenize_dataset.py
Output: Tokenized dataset stored in data/processed/tokenized_dataset/.
```
🎓 4️⃣ Fine-Tune the Models on Dream Data
```sh
python scripts/fine_tune_gpt2.py
Output: Fine-tuned model saved in models/fine_tuned_gpt2/.
```
```sh
python scripts/fine_tune_bart.py
Output: Fine-tuned BART model is saved in models/fine_tuned_bart/.

```
📊 5️⃣ Evaluate the Fine-Tuned Model
```sh
python scripts/evaluate_models.py
Output: Evaluation results saved in data/processed/evaluation_results.json.
```
___
### 🛠 Customization & Configuration
Modify hyperparameters in config/gpt2_training_config.yaml before fine-tuning.
and config/bart_training_config.yaml

Example:
```
num_train_epochs: 5
learning_rate: 3e-5
batch_size: 8
```
___
### 📌 Future Improvements

🔹 Use Reinforcement Learning to refine dream interpretations. 

🔹 Add a web-based UI for interactive dream analysis.

🔹 Enable multilingual dream analysis (e.g., Spanish, French).
___
### ❤️ Contributions & Support
Want to improve AI dream interpretation? Feel free to contribute or open an issue! 🚀
Happy Dreaming! ✨🌙💬