# summarizer.py

# Imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from pdf2image import convert_from_path
import pytesseract
import faiss
import os

# === PATH SETUP ===

# OCR setup - Make sure Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust if needed

# Poppler path (adjust this to your actual extracted Poppler 'bin' directory)
poppler_path = r"C:\poppler\poppler-24.08.0\Library\bin"  # Use the correct path

# === OCR FUNCTION ===

def pdf_to_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    full_text = ""

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        full_text += f"\n\n--- Page {i + 1} ---\n{text}"

    return full_text

# === MAIN PIPELINE ===

filename = "R(3).pdf"
ocr_text = pdf_to_text_with_ocr(filename)

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Try loading FAISS index
try:
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded from local.")
except:
    print("⚙️ FAISS index not found, creating new one...")

    # Split the OCR-extracted text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents([Document(page_content=ocr_text)])

    print(f"📄 Total chunks created: {len(chunks)}")

    # Create FAISS index
    dimension = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(chunks)
    vector_store.save_local("faiss_index")
    print("✅ New FAISS index created and saved.")

# Load LLM
llm = OllamaLLM(model="tinyllama")  # You can change model if needed

# Prompt template
template = """
You are an assistant summarizing sections of a long rulebook.

1. Use only the provided context to answer.
2. Summarize information based on the focus in the question.
3. Do NOT make up any details not present in the context.
4. Return a concise summary in 40 to 50 sentences.

Context:
{context}

Question:
{question}

Summary:
"""
prompt = ChatPromptTemplate.from_template(template)
doc_chain = create_stuff_documents_chain(llm, prompt)

# Define your question
query = "What is geophysical uncertainity according to the pdf?"

# Search top-k similar chunks
results = vector_store.similarity_search(query, k=40)

# Print retrieved sources
print("\n🔍 Retrieved Sources:\n")
for i, doc in enumerate(results):
    print(f"\n--- Source {i+1} ---")
    print(doc.page_content[:1000])  # Print first 1000 chars
    print(f"Metadata: {doc.metadata}")

# Generate response
response = doc_chain.invoke({
    "context": results,
    "question": query
})

# Show the summary
print("\n📘 Final Summary:\n")
print(response)
