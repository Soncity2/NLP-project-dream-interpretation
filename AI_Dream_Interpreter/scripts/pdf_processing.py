import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# Directories
RAW_PDF = Path("../data/raw_pdfs/dreams.pdf")
PROCESSED_DIR = Path("../data/processed")
OUTPUT_FILE = PROCESSED_DIR / "processed_dreams.txt"

print(PROCESSED_DIR)

# Ensure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load Falcon 7B Model
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="cuda:0")

# Define text-generation pipeline
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def extract_text_from_pdf(pdf_path, max_tokens=3000):
    """Extracts raw text from a PDF and limits token length."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    full_text = "\n".join([doc.page_content for doc in documents])

    # Clean unnecessary symbols, extra spaces, and page numbers
    cleaned_text = re.sub(r"\n+", "\n", full_text)  # Remove excessive newlines
    cleaned_text = re.sub(r"\d+\s+", "", cleaned_text)  # Remove numbers that could be page numbers

    # Truncate to max tokens
    tokenized_text = tokenizer(cleaned_text.strip(), truncation=True, max_length=max_tokens, return_tensors="pt")

    return tokenizer.decode(tokenized_text["input_ids"][0], skip_special_tokens=True)


def structure_dream_interpretations(raw_text, max_input_tokens=512, max_output_tokens=200):
    """Forces Falcon 3B-1B to extract OR generate dream interpretations from raw text."""

    # Truncate input text to avoid exceeding model limits
    tokenized_input = tokenizer(raw_text, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    truncated_text = tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

    # 🔹 Stronger few-shot prompt to force structured generation
    prompt = (
        "Read the following text and generate meaningful dream interpretations.\n"
        "If a dream is not explicitly mentioned, create a relevant dream scenario based on the psychology of dreams.\n"
        "Ensure each dream has a clear description and a valid interpretation.\n"
        "Format strictly as follows:\n"
        "\n"
        "Dream: Falling from a high place\n"
        "Interpretation: Fear of failure, losing control, or experiencing instability in life.\n"
        "\n"
        "Dream: Being chased by an unknown figure\n"
        "Interpretation: Anxiety or avoidance of real-life problems.\n"
        "\n"
        "Dream: Losing one's teeth\n"
        "Interpretation: Anxiety about personal appearance or fear of losing control.\n"
        "\n"
        "Now process the following text in the same format:\n"
        f"{truncated_text}\n\n"
        "Extract dreams and interpretations:\n"
    )

    response = generator(prompt, max_new_tokens=max_output_tokens, truncation=True, do_sample=True)

    # ✅ Remove unwanted artifacts
    generated_text = response[0]["generated_text"]
    generated_text = generated_text.replace("<|assistant|>", "").strip()

    # ✅ Extract only "Dream: Interpretation" pairs using regex
    dream_pattern = r"(Dream: .*?)(Interpretation: .*?)"
    extracted_dreams = re.findall(dream_pattern, generated_text, re.DOTALL)

    # ✅ Ensure proper structured output
    if extracted_dreams:
        cleaned_output = "\n\n".join(["\n".join(pair) for pair in extracted_dreams])
    else:
        cleaned_output = "No structured dreams found. The input may not contain extractable dream interpretations."

    return cleaned_output



def split_text_into_chunks(structured_text, chunk_size=512, chunk_overlap=50):
    """Splits structured text into smaller trainable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(structured_text)


def process_pdfs():
    """Processes the PDF, extracts structured dream interpretations, and saves them for fine-tuning."""
    raw_text = extract_text_from_pdf(RAW_PDF)
    structured_text = structure_dream_interpretations(raw_text)
    split_docs = split_text_into_chunks(structured_text)

    # Save structured text
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for text in split_docs:
            f.write(text + "\n\n")

    print(f"Processed text saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_pdfs()
