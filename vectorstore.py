import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Chroma Vector Store with correct embedding model
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"))

# Setup logging
logging.basicConfig(level=logging.INFO)

# Function to load and chunk a PDF document
def load_and_chunk_pdf(pdf_path, chunk_size=1000, overlap=100):
    try:
        logging.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Use RecursiveCharacterTextSplitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = text_splitter.split_documents(documents)

        logging.info(f"PDF loaded and split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logging.error(f"Error loading PDF: {str(e)}")
        return []

# Function to store documents in the vector store
def store_documents_in_vectorstore(documents):
    """Store documents into the vectorstore after chunking them."""
    try:
        if not documents:
            logging.warning("No documents found to store in the vectorstore.")
            return
        
        vectorstore.add_documents(documents)
        logging.info(f"Stored {len(documents)} chunks in the vector store.")
    except Exception as e:
        logging.error(f"Error storing documents: {str(e)}")
        raise

# Function to check if vectorstore contains any documents
def check_vectorstore():
    """Check if the vectorstore contains any documents."""
    try:
        docs = vectorstore.get()
        num_docs = len(docs["documents"]) if "documents" in docs else 0
        logging.info(f"Vectorstore contains {num_docs} documents.")
    except Exception as e:
        logging.error(f"Error checking vectorstore: {str(e)}")

# Function to perform similarity search
def similarity_search(query, k=3):
    try:
        response = openai.embeddings.create(input=[query], model="text-embedding-ada-002")
        query_embedding = response.data[0].embedding  # Corrected access

        # Perform similarity search
        similar_docs = vectorstore.similarity_search_by_vector(query_embedding, k=k)
        
        if not similar_docs:
            logging.warning(f"No relevant documents found for query: {query}")
        
        return similar_docs
    except Exception as e:
        logging.error(f"Error during similarity search: {str(e)}")
        raise

# MAIN EXECUTION: Load PDF, process it, and store in vector DB
if __name__ == "__main__":
    pdf_path = "data/Final BNS Data.pdf"
    documents = load_and_chunk_pdf(pdf_path)
    store_documents_in_vectorstore(documents)
    check_vectorstore()
