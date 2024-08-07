import streamlit as st
import tempfile
import requests
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

# API configuration
API_URL = "" 
API_KEY = ""

# Title and description
st.title(" 💬 Chat with 🦙 LLAMA 3.1 8B on your personal PDF file ")
st.write("Upload a PDF and ask questions about its content.")

# PDF file upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# User input for question
user_question = st.text_input("Ask a question about the PDF content:")

# Use session state to store vectorstore, and other reusable objects
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.pages = None

def call_llama_api(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 120  # Adjust as needed
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json().get('text', 'No response text')
    else:
        return f"Error: {response.status_code} - {response.text}"

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    if st.session_state.pages != pages:
        # Create a vector store using the content from PDF
        vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=None)  # No embedding required for API use
        retriever = vectorstore.as_retriever()

        # Save to session state
        st.session_state.vectorstore = vectorstore
        st.session_state.pages = pages

if user_question:
    if st.session_state.vectorstore is not None:
        template = """
        Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        # Generate prompt and call API
        context = "\n".join([page.page_content for page in st.session_state.pages])
        full_prompt = prompt.format(context=context, question=user_question)
        response = call_llama_api(full_prompt)

        # Display PDF content in the sidebar
        st.sidebar.write("### PDF Content")
        for page in st.session_state.pages:
            st.sidebar.write(page.page_content)

        # Display the response
        st.write("### Response from LLM")
        st.write(response)
