import re
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
dreams_file_path = "common-dreams-psychology.md"
freud_pdf_path = "dreams.pdf"
output_file_path = "dreams_with_freudian_interpretations_structured.txt"

freud_loader = PyMuPDFLoader(freud_pdf_path)
freud_documents = freud_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
freud_chunks = text_splitter.split_text("\n".join([doc.page_content for doc in freud_documents]))
embedding_model = OpenAIEmbeddings()
vector_db = FAISS.from_texts(freud_chunks, embedding_model)

dream_loader = TextLoader(dreams_file_path)
dream_documents = dream_loader.load()
raw_dream_text = dream_documents[0].page_content
dream_chunks = [d.strip() for d in raw_dream_text.split("\n") if d.strip() and not d.startswith(("#", "- "))]

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def get_freud_interpretation(dream):
    search_results = vector_db.similarity_search(dream, k=2)
    if search_results:
        interpretation = " ".join([res.page_content for res in search_results])[:500]
        return clean_text(interpretation)
    return "No clear Freudian interpretation found."

def process_pdfs():
    dream_interpretation_pairs = []
    skipped = 0
    for dream in dream_chunks:
        dream = clean_text(dream)
        if len(dream) < 10:
            skipped += 1
            continue
        interpretation = get_freud_interpretation(dream)
        if "can't be inferred" in interpretation or len(interpretation) < 20:
            skipped += 1
            continue
        dream_interpretation_pairs.append(f"{dream} : {interpretation}")
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(dream_interpretation_pairs))
    logging.info(f"Saved {len(dream_interpretation_pairs)} pairs to {output_file_path}")
    logging.info(f"Skipped {skipped} entries due to length or quality filters.")

if __name__ == "__main__":
    process_pdfs()