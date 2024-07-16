__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import JSONLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


db = Chroma(persist_directory="db", embedding_function=embedding_function)

def query(msg):
    docs = db.similarity_search(msg)
    q = docs[0].page_content
    a = docs[0].metadata
    a = zip(a["Step"].split('|'), a["Answer"].replace('\n', '  \n').split('|'), a["Data"].split('|'), a["Tool"].split('|'))
    return q, a

def main():
    st.set_page_config(page_title="QnA Nghiệp vụ di động", layout="centered")
    st.title("QnA Nghiệp vụ di động")
    msg_input = st.text_area(label="Nội dung tìm kiếm", height=100)
    submit_btn = st.button(label="Submit")
    
    if msg_input:
        if submit_btn:
            with st.spinner("Searching..."):
                q, a = query(msg=msg_input)     
            
            st.write("Câu hỏi thường gặp:  \n" + q)
            for item in a:
                with st.expander(item[0]):
                    st.write("Mẫu câu trả lời:  \n" + item[1])
                    st.write("Dữ liệu:  \n" + item[2])
                    st.write("Công cụ:  \n" + item[3])
                    
            st.success("Done..........")

if __name__ == "__main__":
    main()