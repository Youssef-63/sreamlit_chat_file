import streamlit as st
import tempfile
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from operator import itemgetter

# Access API configuration from Streamlit secrets
API_URL = "https://api.aimlapi.com"
API_KEY = st.secrets["API_KEY"]

# Initialize OpenAI client
openai.api_key = API_KEY
openai.api_base = API_URL

# Title and description
st.title(" 💬 Chat with 🦙 LLAMA 3.1 8B on your personal PDF file ")
st.write("Upload a PDF and ask questions about its content.")

# PDF file upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# User input for question
user_question = st.text_input("Ask a question about the PDF content:")

# Use session state to store PDF pages
if 'pages' not in st.session_state:
    st.session_state.pages = None

def call_llama_api(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant who knows everything."},
                {"role": "user", "content": prompt}
            ]
        )
        message = response.choices[0].message["content"]
        return message
    except Exception as e:
        return f"Error: {str(e)}"

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    # Save to session state
    st.session_state.pages = pages

if user_question:
    if st.session_state.pages is not None:
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
