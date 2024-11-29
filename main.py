import os
import streamlit as st
import pickle
import time
import langchain
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.document_loaders import TextLoader, UnstructuredURLLoader, UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 


google_api_key = os.getenv("google_api_key")
file_path = "vector_index.pkl"
st.title("URL Bot")
st.sidebar.title("News URL")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Enter Url{i+1}")
    urls.append(url)

start_process = st.sidebar.button("Process")
main_placefolder = st.empty()

#initiate llm
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.9, max_tokens=500)

if start_process:
    
    #load the data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("🚀 Data loading... 🧑‍💻 Hang tight, we're almost ready! ⏳")
    data = loader.load()
    # Chunk the data
    splitter = RecursiveCharacterTextSplitter(separators=["\n", " ",  ".", "\n\n"],
                                              chunk_size = 1000,
                                              chunk_overlap = 200
                                              )
    main_placefolder.text("Dissecting the text! 🔍📄")
    chunks = splitter.split_documents(data)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    main_placefolder.text("Building embeddings like a pro, one vector at a time! ⚙️💥")
    vector_index = FAISS.from_documents(chunks, embeddings)
    

    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)

    
query = st.text_input("Question:  ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_stored = pickle.load(f)
            main_placefolder.text("Getting the details for you... 📈🔄 Almost done!")
            chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever=vector_stored.as_retriever())
            result = chain.invoke({'question':query}, return_only_outputs=True)
            st.header("Answer")
            st.text(result['answer'])
            main_placefolder.text("")





    