# --- app.py ---
# This is our main Streamlit application for the RAG chatbot.

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# --- 1. Load Environment Variables ---
load_dotenv()
# We still need the Google API key for the generative model (LLM)
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Please create a .env file with your key.")
    st.stop()

# --- 2. Configure the Streamlit Page ---
st.set_page_config(
    page_title="CV Chatbot - Audrey Dubois",
    page_icon="🤖"
)

# --- 3. Function to Load the Vector Store and Define the RAG Chain ---
# We use a cached function to avoid reloading the model on every interaction.
@st.cache_resource
def get_conversational_chain():
    # Define the prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    # Load the Gemini Pro model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # Create the prompt from the template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Load the Question Answering chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# --- 4. Main App Logic ---
st.header("Chat with Audrey's CV 🤖")

# Load the FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

# Get the user's question
user_question = st.text_input("Ask a question about Audrey's professional experience:")

if user_question:
    # Perform a similarity search in the vector store
    docs = vector_store.similarity_search(user_question)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Get the response from the chain
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    # Display the response
    st.write(response["output_text"])