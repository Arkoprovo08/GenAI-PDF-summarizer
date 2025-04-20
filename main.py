from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains import LLMChain, StuffDocumentsChain , RetrievalQA


filename = "interviewBit_DataScience.pdf"
loader = PyPDFLoader(filename)
pages = loader.load()
text_spliter = SemanticChunker(HuggingFaceEmbeddings())

chunks = text_spliter.split_documents(pages)

print(len(chunks))


for i,chunk in enumerate(chunks):
    print(f"chunk {i}")
    print(chunk)


#embedding
embeddings = HuggingFaceEmbeddings()

d = len(embeddings.embed_query("test query"))
index = faiss.IndexFlatL2(d)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(chunks)


#query
query = "what is overfitting and underfitting ?"


results = vector_store.similarity_search(
    query,
    k=3,
)

print(len(results))


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

print(results)

#ollama

llm = model = OllamaLLM(model="llama3.2")

prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

llm_chain = LLMChain(
                  llm=llm,
                  prompt=QA_CHAIN_PROMPT,
                  callbacks=None,
                  verbose=True)


document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None,
              )


qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True,
              )


print(qa("What is underfitting and over fitting?")["result"])
