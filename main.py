# import os
# import shutil
# from dotenv import load_dotenv
# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq


# load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# def load_documents(upload_folder):
#     documents = []
#     for filename in os.listdir(upload_folder):
#         file_path = os.path.join(upload_folder, filename)
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         elif filename.endswith(".docx"):
#             loader = Docx2txtLoader(file_path)
#         else:
#             continue
#         documents.extend(loader.load())
#     return documents

# def split_documents(documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return text_splitter.split_documents(documents)

# def create_vector_store_from_files(upload_folder):
#     import chromadb
#     if os.path.exists("rag_index"):
#         try:
#             client = chromadb.PersistentClient(path="rag_index")
#             client.reset()
#             shutil.rmtree("rag_index")
#         except Exception as e:
#             print(f"❌ Could not delete old vectorstore: {e}")
#     documents = load_documents(upload_folder)
#     texts = split_documents(documents)
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = Chroma.from_documents(texts, embedding=embeddings, persist_directory="rag_index")
#     # vectorstore.persist()

# def get_rag_chain():
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = Chroma(
#         persist_directory="rag_index",
#         embedding_function=embeddings
#     )
#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 4}
#     )
#     llm = ChatGroq(
#         groq_api_key=os.getenv("GROQ_API_KEY"),
#         model_name="llama3-8b-8192"
#     )
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )


# st.set_page_config(page_title="📄 RAG AI Assistant (PDF/DOCX)")
# st.title("📄 RAG AI Assistant (PDF/DOCX)")

# upload_folder = "uploaded_docs"
# os.makedirs(upload_folder, exist_ok=True)

# uploaded_files = st.file_uploader("📁 Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# if uploaded_files:
#     for file in uploaded_files:
#         with open(os.path.join(upload_folder, file.name), "wb") as f:
#             f.write(file.getbuffer())
#     st.success("✅ Files uploaded.")

#     with st.spinner("⚙️ Creating vector store..."):
#         create_vector_store_from_files(upload_folder)
#         st.success("🧠 Vector store ready!")

# query = st.text_input("🔍 Ask a question based on uploaded documents:")

# if query:
#     if not uploaded_files:
#         st.warning("⚠️ Please upload and process files first.")
#     else:
#         chain = get_rag_chain()
#         response = chain.invoke({"query": query})

#         if response["result"].strip().lower() in ["", "no relevant information found in the uploaded documents."]:
#             st.error("❌ No relevant information found in the uploaded documents.")
#         else:
#             st.markdown("### 🧠 Answer:")
#             st.write(response["result"])

#             with st.expander("📚 Source Documents"):
#                 for doc in response["source_documents"]:
#                     st.markdown(f"**📄 File:** `{doc.metadata.get('source', 'N/A')}`")
#                     st.code(doc.page_content.strip())



import os
import shutil
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def load_documents(upload_folder):
    documents = []
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue
        documents.extend(loader.load())
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_store_from_files(upload_folder):
    import chromadb
    if os.path.exists("rag_index"):
        try:
            client = chromadb.PersistentClient(path="rag_index")
            client.reset()
            shutil.rmtree("rag_index")
        except Exception as e:
            print(f"❌ Could not delete old vectorstore: {e}")
    documents = load_documents(upload_folder)
    texts = split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vectorstore = Chroma.from_documents(texts, embedding=embeddings, persist_directory="rag_index")

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vectorstore = Chroma(
        persist_directory="rag_index",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

st.set_page_config(page_title=" 🤖 RAG-Powered DocBot ")
st.title("🤖 RAG-Powered DocBot ")

upload_folder = "uploaded_docs"
os.makedirs(upload_folder, exist_ok=True)

uploaded_files = st.file_uploader("📁 Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(upload_folder, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success("✅ Files uploaded.")

    with st.spinner("⚙️ Creating vector store..."):
        create_vector_store_from_files(upload_folder)
        st.success("🧠 Vector store ready!")

query = st.text_input("🔍 Ask a question based on uploaded documents:")

if query:
    if not uploaded_files:
        st.warning("⚠️ Please upload and process files first.")
    else:
        chain = get_rag_chain()
        response = chain.invoke({"query": query})

        if response["result"].strip().lower() in ["", "no relevant information found in the uploaded documents."]:
            st.error("❌ No relevant information found in the uploaded documents.")
        else:
            st.markdown("### 🧠 Answer:")
            st.write(response["result"])

            with st.expander("📚 Source Documents"):
                for doc in response["source_documents"]:
                    st.markdown(f"**📄 File:** `{doc.metadata.get('source', 'N/A')}`")
                    st.code(doc.page_content.strip())
