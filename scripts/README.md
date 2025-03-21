# **Processed Files in GPT-2 & T5 Fine-Tuning Pipeline**

## **📌 Introduction**
In this presentation, we will explore the processed files generated in the GPT-2 & T5 fine-tuning pipeline. These files play a crucial role in training, evaluation, and deployment of the model.

---

## **🔹 Overview of the Pipeline**
1. **Raw Data** (PDFs, Text, etc.) → Extracted using `pdf_processing.py` into `dreams_freudian_structure.txt`
2. **Dataset Preparation** → Processed into `dream_interpretations.csv`
4. **Fine-Tuning** → Model is trained with processed and raw datasets and saved in `models/`
5. **Evaluation** → Evaluated model performance with `evaluate_models.py`

---

## **📂 Detailed Explanation of Processed Files**

### **1️⃣ `dreams_freudian_structure.txt`**
- 📍 **Location**: `data/processed/`
- 📄 **Content**: Cleaned and formatted text extracted from PDFs using ChatGPT-4. Created from Claude 200 common dreams in psychology.
- 📌 **Purpose**: Acts as the training dataset before tokenization.

✅ **Example Content:**
```plaintext
    Running but getting nowhere : Indicates avoidance of an unresolved fear, repressed guilt, or internal conflict.
    Paralysis/inability to move : Reflects feelings of helplessness, repression, or unconscious anxiety about control.
```

---

### **2️⃣ `dreams_interpretations.csv`**
- 📍 **Location**: `data/processed/`, `data/raw_pdfs/`
- 📄 **Content**: Processed txt file to csv and raw csv with dreams and their interpretations
- 📌 **Purpose**: Datasets that are used for training the 2 models. One Kaggle Dataset and One Processed from PDF.


✅ **Examples Tokenized Output:**
```csv
Dream,Interpretation
Falling through space,"Symbolizes a loss of control, insecurity, or fear of failure, often linked to anxiety and repressed fears."
Flying effortlessly,"Represents a desire for freedom, escape from constraints, or unconscious sexual excitement."
```

```csv
Dream,Interpretation
"Barbie Doll","To see a Barbie doll in your dream represents society's ideals.  You may feel that you are unable to meet the expectations of others.  Alternatively, the Barbie doll refers to the desire to escape from daily responsibilities. It may serve to bring you back to your childhood where life was much simpler and more carefree."
Barcode,"To see a barcode in your dream symbolizes automation, simplification and ease. Alternatively, the dream represents an impersonal relationship in your waking  life. You are feeling alienated."

```

---

### **3️⃣ `models/` (Fine-Tuned Model Output)**
- 📍 **Location**: `models/fine_tuned_gpt2/`, `models/fine_tuned_t5/`
- 📄 **Content**: Trained 2 model weights and tokenizers.
- 📌 **Purpose**: Stores the fine-tuned model for later use.

✅ **Files Inside `models/`**
- `pytorch_model.bin` → Model weights (large file!)
- `config.json` → Model configuration
- `tokenizer.json` → Tokenizer settings
- `special_tokens_map.json` → Defines special tokens (e.g., `<PAD>`)

---
