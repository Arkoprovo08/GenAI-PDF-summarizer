# summarizer.py (RAG-Optimized with Embedchain-inspired improvements)

# === Imports ===
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

# === Paths ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\poppler\poppler-24.08.0\Library\bin"
pdf_file_path = "R(3).pdf"

# === OCR FUNCTION ===
def pdf_to_text_with_pagewise_docs(pdf_path):
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    documents = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        cleaned = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        documents.append(Document(page_content=cleaned, metadata={"page": i + 1}))
    return documents

# === CHUNKING FUNCTION ===
def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# === EMBEDDING & VECTORSTORE SETUP ===
def setup_vectorstore(chunks, embeddings):
    dimension = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dimension)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vectorstore.add_documents(chunks)
    vectorstore.save_local("faiss_index")
    return vectorstore

# === MAIN ===
print("🔍 Extracting text from PDF...")
docs = pdf_to_text_with_pagewise_docs(pdf_file_path)

print(f"📄 Extracted {len(docs)} pages.")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS or build it
try:
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("✅ Loaded FAISS index from local storage.")
except:
    print("⚙️ FAISS index not found. Rebuilding index...")
    chunks = chunk_documents(docs)
    print(f"🧩 Created {len(chunks)} chunks.")
    vectorstore = setup_vectorstore(chunks, embeddings)
    print("✅ FAISS index created and saved.")

# === LLM SETUP ===
llm = OllamaLLM(model="tinyllama")

# === PROMPT TEMPLATE ===
template = """
You are an assistant helping extract information from PDF documents.
Use only the provided context to answer the question.
Do not guess. Do not infer anything not clearly stated.

If the answer includes values, extract them precisely as stated.
If the answer cannot be found, reply with "Not found in the document."

CONTEXT:
{context}

QUESTION:
{question}

PRECISE ANSWER:
"""
prompt = ChatPromptTemplate.from_template(template)
doc_chain = create_stuff_documents_chain(llm, prompt)

# === QUERYING ===
query = "Fluid Density for wells drilled with SOBM"
print(f"\n🔎 Query: {query}\n")

results = vectorstore.max_marginal_relevance_search(query, k=10)

print("\n🔍 Top Retrieved Chunks:\n")
for i, doc in enumerate(results):
    print(f"\n--- Chunk {i+1} (Page {doc.metadata.get('page', '?')}) ---")
    print(doc.page_content[:400], "\n...")

# Final answer
response = doc_chain.invoke({
    "context": results,
    "question": query
})

print("\n📘 Final Answer:\n")
print(response)
