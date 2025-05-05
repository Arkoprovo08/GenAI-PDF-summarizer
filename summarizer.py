# Imports
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Use for large PDFs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_experimental.text_splitter import SemanticChunker


# Load Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load or Create FAISS Index
try:
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded from local.")
except:
    print("⚙️ FAISS index not found, creating new one...")

    filename = "monopoly.pdf"  
    loader = PyPDFLoader(filename)
    pages = loader.load()

    # Use Recursive splitter (faster for large PDFs than semantic ones)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200
    # )
    text_spliter = SemanticChunker(embeddings)
    chunks = text_spliter.split_documents(pages)

    print(f"📄 Total chunks created: {len(chunks)}")

    # Initialize FAISS index
    dimension = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Add chunks to FAISS and save
    vector_store.add_documents(chunks)
    vector_store.save_local("faiss_index")
    print("✅ New FAISS index created and saved.")

llm = OllamaLLM(model="tinyllama")  

template = """
You are an assistant summarizing sections of a long rulebook.

1. Use only the provided context to answer.
2. Summarize information based on the focus in the question.
3. Do NOT make up any details not present in the context.
4. Return a concise summary in 10 to 15 sentences.

Context:
{context}

Question:
{question}

Summary:
"""
prompt = ChatPromptTemplate.from_template(template)

doc_chain = create_stuff_documents_chain(llm, prompt)

# query = "Summarize all rules related to going to jail and getting out of jail in Monopoly"
query = "Summarize all rules related to Monopoly"

results = vector_store.similarity_search(query, k=15)

# 🔍 Print Retrieved Sources
print("\n🔍 Retrieved Sources:\n")
for i, doc in enumerate(results):
    print(f"\n--- Source {i+1} ---")
    print(doc.page_content[:1000])  # Limit to 1000 chars
    print(f"Metadata: {doc.metadata}")

response = doc_chain.invoke({
    "context": results,
    "question": query
})

print("\n Summary:\n")
print(response)

# with open("summary_output.txt", "w") as f:
#     f.write(str(response))
