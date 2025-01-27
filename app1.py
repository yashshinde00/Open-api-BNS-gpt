import chainlit as cl
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from google.cloud import language_v1
import os

# Load Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Google Cloud NLP client
def init_google_client():
    client = language_v1.LanguageServiceClient()
    return client

# Function to generate embeddings using Google NLP API
def generate_google_embeddings(client, text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_syntax(document=document)
    # Simplistic embedding generation: Use the sentence offsets for now.
    # You can customize this further based on your embedding strategy.
    embeddings = [sentence.text.begin_offset for sentence in response.sentences]
    return embeddings

# Step 1: Load PDFs and create FAISS index
def load_and_index_pdfs(client):
    # Load PDFs from the data folder
    pdf1_loader = PyPDFLoader("data/pdf1.pdf")
    pdf2_loader = PyPDFLoader("data/pdf2.pdf")

    # Extract pages
    pages_pdf1 = pdf1_loader.load()
    pages_pdf2 = pdf2_loader.load()

    # Combine pages
    all_pages = pages_pdf1 + pages_pdf2

    # Generate embeddings using Google NLP API
    documents = []
    for page in all_pages:
        embeddings = generate_google_embeddings(client, page.page_content)
        page.embedding = embeddings
        documents.append(page)

    # Create FAISS index
    faiss_index = FAISS.from_documents(documents)
    return faiss_index

# Step 2: Load Mistral model and tokenizer
def load_mistral_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
    model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", device_map="auto")
    return tokenizer, model

# Chainlit app
@cl.on_chat_start
async def start():
    # Initialize Google Cloud NLP client
    client = init_google_client()

    # Load FAISS index and Mistral model
    faiss_index = load_and_index_pdfs(client)
    tokenizer, model = load_mistral_model()

    # Store in user session
    cl.user_session.set("faiss_index", faiss_index)
    cl.user_session.set("tokenizer", tokenizer)
    cl.user_session.set("model", model)

    # Send a welcome message
    await cl.Message(content="Hello! I'm your legal assistant. Ask me anything about the documents.").send()

@cl.on_message
async def main(message: str):
    # Retrieve FAISS index and Mistral model from user session
    faiss_index = cl.user_session.get("faiss_index")
    tokenizer = cl.user_session.get("tokenizer")
    model = cl.user_session.get("model")

    # ===== RAG: Retrieval =====
    # Retrieve relevant documents using FAISS
    similar_docs = faiss_index.similarity_search(message, k=3)  # Retrieve top 3 relevant documents
    context = "\n\n".join([doc.page_content for doc in similar_docs])  # Combine into a single context

    # ===== RAG: Generation =====
    # Generate response using Mistral with the retrieved context
    input_text = (
        f"Context: {context}\n\n"  # Include retrieved context
        f"Question: {message}\n\n"  # Include user's question
        "Answer:"  # Prompt for the answer
    )

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_length=300,  # Adjust based on your needs
        do_sample=True,  # Enable sampling for diverse responses
        temperature=0.7,  # Control randomness (lower = more deterministic)
        top_p=0.9,  # Nucleus sampling (focus on high-probability tokens)
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Send the response to the frontend
    await cl.Message(content=response).send()
