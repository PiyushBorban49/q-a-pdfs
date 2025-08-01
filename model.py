from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from datasets import load_dataset
import cassio
import PyPDF2
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
import streamlit as st


# All Paths
GEMINI_API="AIzaSyBuXKPw4JCNj9Gx7OWOtMSxpmSvZ-HRGzU"
ASTRA_DB_ID="34b7001d-9dc8-442f-b542-fa078662ae2a"
ASTRA_DB_APPLICATION_TOKEN="AstraCS:PkJsEYTbLxvbuCmKsifPOYNk:f613ec4e87c67f05693b831c11941301153fdcb1351483493e4d84e6cee329cf"


# Read Pdf
pdf = PdfReader(r"/Langchain/Health Insurence Check/90ade7e39d5e481f9aeb772a19a30234.pdf")

# extracting text
raw_text = ""
for pages in pdf.pages:
    raw_text+=pages.extract_text()+'\n'

# initialize
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)

# creating model
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro",google_api_key=GEMINI_API)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GEMINI_API)


# vector storage
astra_vector_store = Cassandra(
    embedding = embedding,
    table_name="context_space",
    session=None,
    keyspace=None
)

# text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# add text in db
astra_vector_store.add_texts(texts)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

st.title("Enter Your Question:")
input_text = st.text_input("Enter Your Question")

if input_text:
    answer = astra_vector_index.query(input_text,llm=llm).strip()
    for doc,score in astra_vector_store.similarity_search_with_score(input_text,k=1):
        st.write(score,doc.page_content)


