# import chainlit as cl
# from langchain_community.document_loaders import PyPDFLoader  # Updated import
# from langchain_community.vectorstores import FAISS  # Updated import
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from fastembed import TextEmbedding  # FastEmbed for embeddings
# import os
# import re

# # Path to save the FAISS index
# FAISS_INDEX_DIR = "faiss_index"

# # Step 1: Preprocess PDFs
# def preprocess_text(text):
#     # Remove headers, footers, and page numbers
#     text = re.sub(r"Page \d+", "", text)  # Remove page numbers
#     text = re.sub(r"\n+", "\n", text)  # Remove extra newlines
#     text = text.strip()  # Remove leading/trailing whitespace
#     return text

# def chunk_text(text, chunk_size=500, overlap=50):
#     # Split text into chunks with overlap
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i : i + chunk_size])
#         chunks.append(chunk)
#     return chunks

# # Step 2: Load PDFs, preprocess, and create FAISS index
# def load_and_index_pdfs():
#     # Check if FAISS index already exists
#     if os.path.exists(FAISS_INDEX_DIR):
#         print("Loading existing FAISS index...")
#         return FAISS.load_local(FAISS_INDEX_DIR, embeddings=TextEmbedding())

#     # Load PDFs from the data folder
#     pdf1_loader = PyPDFLoader("data/pdf1.pdf")
#     pdf2_loader = PyPDFLoader("data/pdf2.pdf")

#     # Extract pages
#     pages_pdf1 = pdf1_loader.load()
#     pages_pdf2 = pdf2_loader.load()

#     # Combine pages
#     all_pages = pages_pdf1 + pages_pdf2

#     # Preprocess and chunk text
#     chunks = []
#     for page in all_pages:
#         cleaned_text = preprocess_text(page.page_content)
#         page_chunks = chunk_text(cleaned_text)
#         chunks.extend(page_chunks)

#     # Generate embeddings using FastEmbed
#     embedding_model = TextEmbedding()  # Initialize FastEmbed
#     embeddings = list(embedding_model.embed(chunks))  # Generate embeddings

#     # Create FAISS index
#     faiss_index = FAISS.from_embeddings(
#         text_embeddings=list(zip(chunks, embeddings)),
#         embedding=embedding_model,  # Pass the embedding model
#     )

#     # Save FAISS index locally
#     faiss_index.save_local(FAISS_INDEX_DIR)
#     print("FAISS index saved locally.")
#     return faiss_index

# # Step 3: Load Mistral model and tokenizer
# def load_mistral_model():
#     # Load tokenizer and model directly
#     tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
#     model = AutoModelForCausalLM.from_pretrained(
#         "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
#         trust_remote_code=True,  # Required for GPTQ models
#     )
#     return tokenizer, model

# # Chainlit app
# @cl.on_chat_start
# async def start():
#     # Load FAISS index and Mistral model
#     faiss_index = load_and_index_pdfs()
#     tokenizer, model = load_mistral_model()

#     # Store in user session
#     cl.user_session.set("faiss_index", faiss_index)
#     cl.user_session.set("tokenizer", tokenizer)
#     cl.user_session.set("model", model)

#     # Send a welcome message
#     await cl.Message(content="Hello! I'm your legal assistant. Ask me anything about the documents.").send()

# @cl.on_message
# async def main(message: str):
#     # Retrieve FAISS index and Mistral model from user session
#     faiss_index = cl.user_session.get("faiss_index")
#     tokenizer = cl.user_session.get("tokenizer")
#     model = cl.user_session.get("model")

#     # ===== RAG: Retrieval =====
#     # Retrieve relevant documents using FAISS
#     similar_docs = faiss_index.similarity_search(message, k=3)  # Retrieve top 3 relevant documents
#     context = "\n\n".join([doc.page_content for doc in similar_docs])  # Combine into a single context

#     # ===== RAG: Generation =====
#     # Generate response using Mistral with the retrieved context
#     input_text = (
#         f"Context: {context}\n\n"  # Include retrieved context
#         f"Question: {message}\n\n"  # Include user's question
#         "Answer:"  # Prompt for the answer
#     )

#     # Tokenize input
#     inputs = tokenizer(input_text, return_tensors="pt")

#     # Generate response
#     outputs = model.generate(
#         **inputs,
#         max_length=300,  # Adjust based on your needs
#         do_sample=True,  # Enable sampling for diverse responses
#         temperature=0.7,  # Control randomness (lower = more deterministic)
#         top_p=0.9,  # Nucleus sampling (focus on high-probability tokens)
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Send the response to the frontend
#     await cl.Message(content=response).send()   