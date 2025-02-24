# **AI Dream Interpreter** 🌙💭✨  
**Fine-tune LLaMA 2 to interpret dreams based on structured text from PDFs.**  

---

## 🌟 **Overview**  
This project fine-tunes **LLaMA 2** using dream interpretation texts extracted from PDFs. The fine-tuned model can **analyze and interpret dreams** based on input prompts.  

### **Pipeline:**  
✅ Extract text from PDFs 📄  
✅ Preprocess & tokenize the dataset 🛠️  
✅ Fine-tune LLaMA 2 on dream-related data 🧠  
✅ Evaluate model accuracy and performance 📊  

---

## 🚀 **Setup Guide**  

### **1️⃣ Clone the Repository**  
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
│   ├── training_config.yaml       # Training hyperparameters
│── data/                         
│   ├── raw_pdfs/                  # PDF files for fine-tuning
│   ├── processed/                  # Extracted & processed text
│── models/                        # Trained models
│── scripts/                      
│   ├── pdf_processing.py           # Extract text from PDFs
│   ├── dataset_preparation.py      # Prepare dataset
│   ├── tokenize_dataset.py         # Tokenize text
│   ├── fine_tune.py                # Fine-tune LLaMA 2
│   ├── evaluate.py                 # Evaluate model
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
Output: A structured dataset in data/processed/dataset.jsonl.
```
🔢 3️⃣ Tokenize the Dataset
```sh
python scripts/tokenize_dataset.py
Output: Tokenized dataset stored in data/processed/tokenized_dataset/.
```
🎓 4️⃣ Fine-Tune LLaMA 2 on Dream Data
```sh
python scripts/fine_tune.py
Output: Fine-tuned model saved in models/fine_tuned_llama2/.
```
📊 5️⃣ Evaluate the Fine-Tuned Model
```sh
python scripts/evaluate_models.py
Output: Evaluation results saved in data/processed/evaluation_results.json.
```
___
### 🛠 Customization & Configuration
Modify hyperparameters in config/training_config.yaml before fine-tuning.

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