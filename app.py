__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import JSONLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
import json

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


db = Chroma(persist_directory="db", embedding_function=embedding_function)

def query(msg):
    docs = db.similarity_search(msg)
    return docs[0].page_content, docs[0].metadata

def main():
    st.set_page_config(page_title="QnA Nghiệp vụ di động", layout="centered")
    st.title("QnA Nghiệp vụ di động")
    msg_input = st.text_area(label="Nội dung tìm kiếm", height=100)
    submit_btn = st.button(label="Submit")
    
    if msg_input:
        if submit_btn:
            result = query(msg=msg_input)
            st.text_area(label="Q", value=result[0], height=100)
            st.text_area(label="A", value=result[1], height=300)

            

if __name__ == "__main__":
    main()
#msg = "nạp tiền ví"
#print(str(msg.encode('utf-8')))

#obj = query(msg=msg)
#print(obj[0])