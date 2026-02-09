from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

docs = []

for file in os.listdir("data/docs"):
    loader = PyPDFLoader(f"data/docs/{file}")
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="vector_db"
)


print("Ingestion complete successfully")
