import streamlit as st
import tempfile
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

# Title and description
st.title(" 💬 Chat with 🦙 ollama on your personal PDF file ")
st.write("Upload a PDF and ask questions about its content.")

# PDF file upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# User input for question
user_question = st.text_input("Ask a question about the PDF content:")

# Use session state to store embeddings, vectorstore, and other reusable objects
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.embeddings = None
    st.session_state.pages = None

if uploaded_file:
    model = "llama3.1:8b"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    if st.session_state.pages != pages:
        embeddings = OllamaEmbeddings(model=model)
        vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Save to session state
        st.session_state.vectorstore = vectorstore
        st.session_state.embeddings = embeddings
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
        model = OllamaLLM(model=model)
        parser = StrOutputParser()

        chain = (
            {
                "context": itemgetter("question") | st.session_state.vectorstore.as_retriever(),
                "question": itemgetter("question"),
            }
            | prompt
            | model
            | parser
        )

        response = chain.invoke({"question": user_question})

        # Display PDF content in the sidebar
        st.sidebar.write("### PDF Content")
        for page in st.session_state.pages:
            st.sidebar.write(page.page_content)

        # Display the response
        st.write("### Response from LLM")
        st.write(response)


