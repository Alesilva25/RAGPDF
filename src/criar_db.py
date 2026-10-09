from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = 'base'
DB_PATH = 'db'

def create_db():
    documents = load_documents()
    chunks = split_chunks(documents)
    vetorized_chunks(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader(BASE_DIR, glob="*.pdf")
    documents = loader.load()

    return documents

def split_chunks(documents):
    documents_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = documents_splitter.split_documents(documents)
    print(len(chunks))

    return chunks

def vetorized_chunks(chunks):
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=DB_PATH
    )

    print("Banco de dados criado")

create_db()