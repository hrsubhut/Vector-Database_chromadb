import os
import shutil
import zipfile
import urllib.request
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# =====================================================================
# 1. ENVIRONMENT CONFIGURATION
# =====================================================================
# SECURE CONFIGURATION: Drop your OpenAI Secret API key here
os.environ["OPENAI_API_KEY"] = "ENTER_YOUR_OPENAI_API_KEY_HERE"

# =====================================================================
# 2. DATA INGESTION & EXTRACTION
# =====================================================================
DATA_URL = "[https://www.dropbox.com/scl/fi/7uxgawpntssj84j94j7g0/new_articles.zip?rlkey=5nux0q2smykplxlugr2o63056&dl=1](https://www.dropbox.com/scl/fi/7uxgawpntssj84j94j7g0/new_articles.zip?rlkey=5nux0q2smykplxlugr2o63056&dl=1)"
ZIP_FILE = "new_articles.zip"
EXTRACT_DIR = "new_articles"
PERSIST_DIR = "db"

print("Downloading document archive...")
urllib.request.urlretrieve(DATA_URL, ZIP_FILE)

print("Extracting files...")
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

# Loading target text files into memory
print("Ingesting raw text artifacts...")
loader = DirectoryLoader(EXTRACT_DIR, glob="./*.txt", loader_cls=TextLoader)
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} source files successfully.")

# =====================================================================
# 3. TEXT SPLITTING (CHUNKING)
# =====================================================================
print("Executing text fragmentation...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
document_chunks = text_splitter.split_documents(raw_documents)
print(f"Generated {len(document_chunks)} discrete context chunks.")

# =====================================================================
# 4. CHROMADB VECTORIZATION & LOCAL STORAGE
# =====================================================================
print("Initializing embedding models and populating ChromaDB...")
embedding_model = OpenAIEmbeddings()

# Build vector store instance from documents
vectordb = Chroma.from_documents(
    documents=document_chunks,
    embedding=embedding_model,
    persist_directory=PERSIST_DIR
)

# Persist DB metadata locally to disk
vectordb.persist()
print(f"Vector Database safely saved to local disk path: '{PERSIST_DIR}'")

# =====================================================================
# 5. RETRIEVAL QA PIPELINE EXECUTION
# =====================================================================
print("Configuring search retriever metadata...")
# Setup the search interface to fetch the top 2 matching vectors
search_retriever = vectordb.as_retriever(search_kwargs={"k": 2})

print("Constructing Retrieval QA Chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=search_retriever,
    return_source_documents=True
)

# Pipeline verification query execution
test_query = "How much money did Microsoft raise?"
print(f"\nProcessing Pipeline Query: '{test_query}'")

execution_response = qa_chain.invoke({"query": test_query})

# Display formatted generation output
print("\n=================== PIPELINE GENERATION ANSWER ===================")
print(execution_response["result"].strip())

print("\n=================== REFERENCED CONTENT SOURCES ===================")
for ranking, document in enumerate(execution_response["source_documents"], start=1):
    print(f"[{ranking}] File Reference: {document.metadata.get('source')}")
