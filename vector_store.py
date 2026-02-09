from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def load_db():
    return Chroma(
        persist_directory="vector_db",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
