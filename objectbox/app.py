
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_objectbox.vectorstores import ObjectBox

# Environment variable keys
groq_api_key = os.environ['groq_api_key']
gemini_api_key = os.environ['google_api_key']

st.title("ObjectBox Vector DB with Groq llam3")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Vector Embedding and Objectbox Vectorstore db
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.docs = PyPDFDirectoryLoader("./data").load()  # Doc ingestion and loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(gemini_api_key=gemini_api_key, model="models/embedding-001")
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=768)

input_prompt = st.text_input("Enter your Question From Doc")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("ObjectBox Database is Ready")

import time

if input_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    if "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()

        response = retrieval_chain.invoke({'input': input_prompt})

        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Please initialize the vectors by clicking 'Document Embedding' button first.")
