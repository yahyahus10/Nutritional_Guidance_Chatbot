
## **Loading_PDF and extracting data **

from langchain_community.document_loaders import PyPDFDirectoryLoader

#This defines a Python function named load_pdf, which takes one parameter called data.
#The purpose of this function is to load PDF files as documents
#The parameter data is expected to be a path to the directory containing the PDF files that need to be loaded
def load_pdf(data):
#an instance of PyPDFDirectoryLoader is created, named loader. This instance is initialized with two arguments:
#The second argument is glob="*.pdf". The glob parameter specifies a pattern that matches files in the directory.
#Here, "*.pdf" indicates that all files ending with .pdf should be considered
    loader = PyPDFDirectoryLoader(data, glob="*.pdf")

    documents=loader.load()

    return documents

## **Loading and extracting data from the web **
from langchain_community.document_loaders import WebBaseLoader

def load_web(url):
    loader= WebBaseLoader(url)
    documents = loader.load()
    return documents

## **Splitting_text into chunks**

from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_split(extracted_data):
  '''#n instance of RecursiveCharacterTextSplitter is created, named text_splitter
  #chunk_size=200: This sets the size of each chunk of text to 200 characters.
  This means that the text will be divided into blocks, each containing up to 200 characters.

  #chunk_overlap=25: This sets the overlap between consecutive chunks to 25 characters.
  This overlap means that each new chunk starts 25 characters before the previous chunk ended'''
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)

  text_chunks = text_splitter.split_documents(extracted_data)
  return text_chunks

## **Download embedding model**

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,)

def download_hugging_face_embeddings():
   embedding= SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   return embedding