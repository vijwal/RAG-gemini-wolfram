import streamlit as st
import os
import google.generativeai as genai
from pathlib import Path as p
import PyPDF2
from PyPDF2 import PdfReader
import pandas as pd
from IPython.display import display, Markdown
import re
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

GOOGLE_API_KEY="AIzaSyD-8SRv-pnoFz9zqMzYIiaDYWaw52qAXvk"
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config("Gemini")
st.header("Wolfram + Gemini")
texts = []
folder_path = r"C:\Users\manoc\Downloads\New folder"

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        print(file)
        try:
            with open(os.path.join(folder_path, file), 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                pages = pdf_reader.pages
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


                problematic_pattern = r"\x0c"
                clean_text = ""
                for page in pages:
                    clean_page_text = re.sub(problematic_pattern, "", page.extract_text())
                    clean_text += clean_page_text + "\n\n"

                context = clean_text
                texts.extend(text_splitter.split_text(context))

        except PyPDF2.errors.PdfReadError as e:
            print(f"Error processing {file}: {e}")
            try:
                repaired_pdf = PyPDF2.PdfReader(os.path.join(folder_path, file)).getPage(0)
                with open(os.path.join(folder_path, "repaired_" + file), 'wb') as output_file:
                    repaired_pdf.write(output_file)
                    print(f"Created a repaired version: {file}")
            except Exception as e:
                print(f"Failed to repair {file}: {e}")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 10})
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY,temperature=0.2,convert_system_message_to_human=True)

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved background context to answer the question as Detailedly as possible. If the answer is not in the 
    provided context or data just say, "answer is not available" and if the answer is math related and you dont know the answer just say "Wolfram"\n\n
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
)

question = st.text_input("Ask a Question")

import wolframalpha 
client = wolframalpha.Client(app_id) 

on = st.toggle('Switch to Wolfram Alpha')
import requests
import xml.etree.ElementTree as ET
url="https://api.wolframalpha.com/v1/query?input={}&format=image,plaintext&output=XML&appid=927UG2-4HTKGJXA65"
import urllib.parse

f_url=url.format(urllib.parse.quote(question, safe=''))

if st.button("Submit"):
        with st.spinner("Processing..."):
            result = qa_chain({"query": question})
            # st.write(result["result"])
            if result["result"] == "wolfram" or result["result"]== "Wolfram" or on:
                response = requests.get(f_url)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    for pod in root.findall(".//pod"):
                        title = pod.attrib.get('title', '')
                        st.write(title)
                        for subpod in pod.findall(".//subpod"):
                            img = subpod.find(".//img")
                            if img is not None:
                                img_url = img.attrib.get('src', '')
                                st.image(img_url)
                            plaintext = subpod.find(".//plaintext")
                            if plaintext is not None:
                                st.write(" -", plaintext.text)
                            else:
                                st.write("Error:", response.status_code)
            elif on == False:
                st.write("Reply: ", Markdown(result["result"]))
