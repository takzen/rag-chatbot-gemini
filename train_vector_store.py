# --- train_vector_store.py ---
# This script loads a document, splits it into chunks, creates embeddings
# using a local Hugging Face model, and stores them in a FAISS vector database.

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # IMPORT CHANGE
from langchain_community.vectorstores import FAISS

print("Starting the vector store creation process...")

# --- 1. Load the Document ---
loader = TextLoader('data/cv_audrey_dubois.txt', encoding='utf-8')
documents = loader.load()
print(f"Loaded {len(documents)} document(s).")

# --- 2. Split the Document into Chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)
print(f"Document split into {len(docs)} chunks.")

# --- 3. Create Embeddings using a Local Hugging Face Model ---
# We are using a popular and powerful open-source model.
# The first time you run this, it will be downloaded to your computer.
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
print("Hugging Face embedding model loaded.")


# --- 4. Build the Vector Store ---
print("Creating vector store with FAISS...")
vector_store = FAISS.from_documents(docs, embeddings)

# --- 5. Save the Vector Store Locally ---
vector_store_path = "vector_store"
vector_store.save_local(vector_store_path)

print(f"Vector store created and saved successfully at: {vector_store_path}")