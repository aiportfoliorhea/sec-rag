from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from constants import CHUNK_SIZE, CHUNK_OVERLAP
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_retriever_cache = None
def load_vector_store():
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache
    
    with open(os.path.join(BASE_DIR, "jpm-10K-small-clean.txt"), "r")  as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)
    _retriever_cache = vector_store.as_retriever()
    return _retriever_cache