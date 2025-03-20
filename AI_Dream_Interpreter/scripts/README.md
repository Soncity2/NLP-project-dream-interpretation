# **Processed Files in LLaMA 2 Fine-Tuning Pipeline**

## **📌 Introduction**
In this presentation, we will explore the processed files generated in the LLaMA 2 fine-tuning pipeline. These files play a crucial role in training, evaluation, and deployment of the model.

---

## **🔹 Overview of the Pipeline**
1. **Raw Data** (PDFs, Text, etc.) → Extracted using `pdf_processing.py`
2. **Dataset Preparation** → Processed into `dataset.txt`
3. **Tokenization** → `tokenized_dataset/` created using `tokenize_dataset.py`
4. **Fine-Tuning** → Model is trained and saved in `models/`
5. **Evaluation** → Evaluated model performance with `evaluate.py`

---

## **📂 Detailed Explanation of Processed Files**

### **1️⃣ `dataset.txt`**
- 📍 **Location**: `data/processed/dataset.txt`
- 📄 **Content**: Cleaned and formatted text extracted from PDFs or other sources.
- 📌 **Purpose**: Acts as the training dataset before tokenization.

✅ **Example Content:**
```plaintext
"Dreams are a window into the subconscious."
"Freud's interpretation of dreams revolutionized psychology."
```

---

### **2️⃣ `tokenized_dataset/`**
- 📍 **Location**: `data/processed/tokenized_dataset/`
- 📄 **Content**: Hugging Face Dataset files (binary format) containing tokenized text.
- 📌 **Purpose**: Stores text converted into token IDs, which are used for training the LLaMA 2 model.

✅ **Files Inside `tokenized_dataset/`**
- `data-00000-of-00001.arrow` → Stores tokenized text in Apache Arrow format.
- `dataset_info.json` → Metadata about the dataset.
- `state.json` → Tracks processing status.

✅ **Example Tokenized Output:**
```json
{"text": "Dreams are a window into the subconscious.", "input_ids": [342, 654, 1234, ...], "attention_mask": [1, 1, 1, ...]}
```

---

### **3️⃣ `models/` (Fine-Tuned Model Output)**
- 📍 **Location**: `models/fine_tuned_llama2/`
- 📄 **Content**: Trained LLaMA 2 model weights and tokenizer.
- 📌 **Purpose**: Stores the fine-tuned model for later use.

✅ **Files Inside `models/`**
- `pytorch_model.bin` → Model weights (large file!)
- `config.json` → Model configuration
- `tokenizer.json` → Tokenizer settings
- `special_tokens_map.json` → Defines special tokens (e.g., `<PAD>`)

---

## **📊 Data Flow in Fine-Tuning**
```plaintext
Raw PDFs → dataset.txt → tokenized_dataset/ → Fine-Tuned Model (models/)
```

---

## **🛠 Managing Processed Files in Git**

### **Why Aren’t Processed Files Automatically Tracked?**
1. **They might be in `.gitignore`** → Check & remove exclusions.
2. **Files are too large** → Use `Git LFS` for large model weights.
3. **Some files are dynamically generated** → Avoid pushing temporary files.

### **How to Track Processed Files?**
```bash
git add -f data/processed/dataset.txt
git add -f data/processed/tokenized_dataset/*
git add -f models/*
git commit -m "Adding processed files"
git push origin main
```

For large files:
```bash
git lfs track "models/*"
git add .gitattributes
git commit -m "Using Git LFS for large files"
git push origin main
```

---
