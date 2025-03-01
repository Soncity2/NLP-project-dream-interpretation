import re
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# File paths
dreams_file_path = "common-dreams-psychology.md"  # List of common dreams
freud_pdf_path = "dreams.pdf"  # Freud's book

# Load Freud's text from PDF using LangChain
print("Loading Freud's book...")
freud_loader = PyMuPDFLoader(freud_pdf_path)
freud_documents = freud_loader.load()

# Clean and Split Freud's text for efficient searching
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
freud_chunks = text_splitter.split_text("\n".join([doc.page_content for doc in freud_documents]))

# Create a searchable vector database (FAISS) for Freud’s interpretations
print("Indexing Freud's interpretations...")
embedding_model = OpenAIEmbeddings()  # Requires OpenAI API key
vector_db = FAISS.from_texts(freud_chunks, embedding_model)

# Load and process the list of common dreams
print("Processing common dreams...")
dream_loader = TextLoader(dreams_file_path)
dream_documents = dream_loader.load()
dream_chunks = text_splitter.split_text(dream_documents[0].page_content)

# Define function to find relevant Freud interpretations
def get_freud_interpretation(dream):
    """Searches Freud's book for the most relevant interpretation of a dream."""
    search_results = vector_db.similarity_search(dream, k=1)  # Retrieve top match
    return search_results[0].page_content if search_results else "Interpretation can't be inferred from Freudian dream analysis principles."



def process_pdfs():
    print("Generating interpretations...")
    dream_interpretation_pairs = []
    for dream in dream_chunks:
        dream = dream.strip()
        if dream and not dream.startswith("#") and not dream.isdigit():  # Ignore headings and numbers
            interpretation = get_freud_interpretation(dream)
            dream_interpretation_pairs.append(f"{dream} : {interpretation}")

    # Step 7: Save structured output to a file
    output_file_path = "dreams_with_freudian_interpretations_structured.txt"
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(dream_interpretation_pairs))

    print(f"Structured dream interpretations saved to {output_file_path}")


if __name__ == "__main__":
    process_pdfs()
