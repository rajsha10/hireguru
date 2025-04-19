import os
import sys
from io import StringIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from resume_summary_model import get_response
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def resume_summary(text):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(text)
    context = "\n".join([doc.page_content for doc in docs])
    answer = get_response(context)
    return answer

def main():
    pdf_directory = "resumes"
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    results = {}
    
    try:
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, filename)
                
                raw_text = get_pdf_text(pdf_path)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                summary = resume_summary(raw_text)
                results[filename] = summary
    finally:
        sys.stdout = old_stdout
    
    for filename, summary in results.items():
        print(f"\nSummary for {filename}:")
        print(summary)

if __name__ == "__main__":
    main()