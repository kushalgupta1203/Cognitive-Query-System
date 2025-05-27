import streamlit as st
import os
import textwrap
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import io  # Import io for BytesIO
from docx import Document
import pandas as pd  # Import pandas for spreadsheet handling

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get Gemini response for text input
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(question)
    return response.text

# Function to get Gemini response for image input
def get_gemini_response_image(input, image):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if input != "":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

# Function to get text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))  # Convert bytes to a file-like object
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Safeguard against None values
    return text

# Function to get chunks of text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say, "answer is not available in the context".
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input for PDF questions
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to extract text from a Word document
def extract_text_from_docx(file):
    doc = Document(file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)

# Function to get response from the Gemini model for Word documents
def get_gemini_response_word(input_text, document_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    full_input = f"{input_text}\n\n{document_text}" if input_text else document_text
    response = model.generate_content([full_input])
    return response.text

# Function to read spreadsheet and return its content
def read_spreadsheet(file):
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        return None
    return df  # Return DataFrame for analysis

# Function to get response from the Gemini model for spreadsheets
def get_gemini_response_spreadsheet(input_text, data_frame):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if input_text:
        # Convert the DataFrame to a string to include in the query
        data_summary = data_frame.to_string(index=False)  # Don't include the index
        combined_input = f"{input_text}\n\nData:\n{data_summary}"
        response = model.generate_content([combined_input])
        return response.text
    return "No input provided."

# Initialize the Streamlit app
st.set_page_config(page_title="Cognitive Query System")  # Updated page title
st.title("Cognitive Query System Using Generative AI 🤖")  # Main title

# Sidebar menu
st.sidebar.title("Menu")
st.sidebar.subheader("Select Query Type")

# Main options for query types
option = st.sidebar.radio("Choose an option:", ["Ask a Question", "Analyze an Image", "Explore a Document"])

# Define functions for each query type
def text_query():
    st.header("Ask Your Queries Using CQS 💬")
    input_text = st.text_input("Input Prompt: ", key="input_text")
    if st.button("Submit"):  # Uniform button text
        response = get_gemini_response(input_text)
        st.subheader("The Response is:")
        st.write(response)

def image_query():
    st.header("Chat with Image Using CQS 🖼️")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])  # Moved up
    image = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)

    input_text = st.text_input("Input Prompt: ", key="input_image")  # Moved down

    if st.button("Submit"):  # Uniform button text
        if image:  # Ensure an image is uploaded before submitting
            response = get_gemini_response_image(input_text, image)
            st.subheader("The Response is:")
            st.write(response)
        else:
            st.warning("Please upload an image before submitting your query.")

def document_query():
    st.header("Chat with Documents Using CQS 📄")

    # Dropdown menu for document type selection
    doc_type = st.selectbox("Select Document Type", ["Select...", "PDF Document", "Word Document", "Spreadsheet Document"])

    # PDF Section
    if doc_type == "PDF Document":
        st.subheader("PDF Documents")
        pdf_docs = st.file_uploader("Upload PDF Files (.pdf)", type=["pdf"], key="pdf_uploader", accept_multiple_files=True)
        user_question_pdf = st.text_input("Input Prompt: ", key="pdf_question")
        if st.button("Submit"):  # Uniform button text
            if user_question_pdf and pdf_docs:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    response = user_input(user_question_pdf)
                    st.subheader("The Response is:")
                    st.write(response)

    # Word Document Section
    elif doc_type == "Word Document":
        st.subheader("Word Documents")
        uploaded_file = st.file_uploader("Choose a Word document...", type=["docx"], key="word_uploader")
        input_prompt = st.text_input("Input Prompt:", key="input_word")

        if uploaded_file is not None:
            # Extract text from the uploaded Word document
            document_text = extract_text_from_docx(uploaded_file)
            st.text_area("Document Content:", value=document_text, height=300)

        if st.button("Submit"):  # Uniform button text
            # Combine the input prompt and document text for analysis
            response = get_gemini_response_word(input_prompt, document_text)
            st.subheader("The Response is:")
            st.write(response)

    # Spreadsheet Document Section
    elif doc_type == "Spreadsheet Document":
        st.subheader("Spreadsheet Documents")
        uploaded_file = st.file_uploader("Choose a spreadsheet file...", type=["xlsx", "csv"], key="spreadsheet_uploader")
        input_prompt = st.text_input("Input Prompt:", key="input_spreadsheet")

        if uploaded_file is not None:
            # Read the uploaded spreadsheet
            spreadsheet_df = read_spreadsheet(uploaded_file)

        if st.button("Submit"):  # Uniform button text
            if uploaded_file is not None and spreadsheet_df is not None:
                # Get the response from the Gemini model for spreadsheet data
                response = get_gemini_response_spreadsheet(input_prompt, spreadsheet_df)
                st.subheader("The Response is:")
                st.write(response)
            else:
                st.error("Error reading the spreadsheet. Please upload a valid file.")

# Conditional rendering based on the selected option
if option == "Ask a Question":
    text_query()
elif option == "Analyze an Image":
    image_query()
elif option == "Explore a Document":
    document_query()
