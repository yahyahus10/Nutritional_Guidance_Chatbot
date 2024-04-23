from src.helper_ import load_pdf, load_web, text_split, download_hugging_face_embeddings
from langchain_chroma import Chroma

Hypertension_document_pdf=load_pdf("Hypertension_data/")
Hypertension_document_web_1 = load_web("https://www.webmd.com/hypertension-high-blood-pressure/high-blood-pressure-diet")
Hypertension_document_web_2=load_web("https://www.nhs.uk/live-well/eat-well/food-guidelines-and-food-labels/the-eatwell-guide/")

extracted_merged_Hpertension_data = Hypertension_document_web_1 + Hypertension_document_pdf + Hypertension_document_web_2

text_chunks=text_split(extracted_merged_Hpertension_data)
embeddings=download_hugging_face_embeddings()

db = Chroma.from_documents(text_chunks, embeddings, persist_directory="./chroma_db")