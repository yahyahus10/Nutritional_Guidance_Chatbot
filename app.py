from flask import Flask, render_template, jsonify, request
from src.helper_ import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.prompt_ import*
from src.helper_ import load_pdf, load_web, text_split, download_hugging_face_embeddings
from langchain_chroma import Chroma

import os

app=Flask(__name__)

Hypertension_document_pdf=load_pdf("Hypertension_data/")
Hypertension_document_web_1 = load_web("https://www.webmd.com/hypertension-high-blood-pressure/high-blood-pressure-diet")
Hypertension_document_web_2=load_web("https://www.nhs.uk/live-well/eat-well/food-guidelines-and-food-labels/the-eatwell-guide/")

extracted_merged_Hpertension_data = Hypertension_document_web_1 + Hypertension_document_pdf + Hypertension_document_web_2
text_chunks=text_split(extracted_merged_Hpertension_data)

embeddings=download_hugging_face_embeddings()

db = Chroma.from_documents(text_chunks, embeddings, persist_directory="./chroma_db")

PROMPT=PromptTemplate(template=promt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt":PROMPT}
'''The inclusion of PROMPT in chain_type_kwargs suggests that the system might use
these configurations to structure its processing chain, where the prompt plays a central role
in interacting with the model.'''

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin", #LaMA model variant with around 2.7 billion parameters, tailored for chat applications
    model_type="llama",
    config={
        'max_new_tokens': 512, #max_new_tokens': 512 defines the maximum number of tokens (words or sub-words) that the model can generate in one invocation
        'temperature': 0.8  # Corrected to proper key-value format
        #temperature': 0.8 controls the creativity or randomness of the output generated by the model.
        #Lower temperatures lead to more predictable, repetitive text, while higher temperatures produce more varied and novel text
    }
)

qa = RetrievalQA.from_chain_type( #reate an instance of a RetrievalQA class
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 4}),#k=2 :Will give two most relevant answer/document
    #This line configures the retriever component of the QA system. db.as_retriever converts a database (db)
    #into a retriever that can search for and retrieve documents relevant to the query.
    return_source_documents=True,  # Corrected spelling
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)