from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains import LLMChain, StuffDocumentsChain , RetrievalQA
from langchain.chains.combine_documents import  create_stuff_documents_chain


# #embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


try:
    vector_store = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
except:
    filename = "monopoly.pdf"
    loader = PyPDFLoader(filename)
    pages = loader.load()
    # text_spliter = SemanticChunker(HuggingFaceEmbeddings())
    text_spliter = SemanticChunker(embeddings)

    chunks = text_spliter.split_documents(pages)

    print(len(chunks))

    d = len(embeddings.embed_query("test query"))
    index = faiss.IndexFlatL2(d)


    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(chunks)
    vector_store.save_local("faiss_index")


# #query
# query = "how to get out of jail?"
# query = "Give a summary of the Monopoly rules."
query = "what is the bank?"

results = vector_store.similarity_search(
    query,
    k=3,
)

print(results)

# #ollama

llm = model = OllamaLLM(model="tinyllama")

template = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.
\n\n
Context: {context}\n\n

Question: {question}\n\n

Helpful Answer:"""

prompt = ChatPromptTemplate.from_template(template)

doc_chain = create_stuff_documents_chain(llm,prompt)

# query = "Give summary of monopoly.pdf"

print(doc_chain.invoke({"context":results,"question":query}))
