import os
import logging
import openai
import chainlit as cl
from dotenv import load_dotenv
from vectorstore import similarity_search

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)

@cl.on_message
async def main(message: cl.Message):
    user_message = message.content

    if len(user_message) > 1000:  # Prevent oversized requests
        await cl.Message(content="⚠️ Message too long! Please shorten it.").send()
        return

    try:
        # Step 1: Retrieve relevant documents
        similar_docs = similarity_search(user_message, k=3)

        if not similar_docs:
            await cl.Message(content="😕 No relevant documents found. Try rephrasing your question or asking about a specific legal topic.").send()
            return

        # Step 2: Prepare context from retrieved documents
        context_text = "\n\n".join([doc.page_content for doc in similar_docs])
        
        # Step 3: Ask GPT-3.5-Turbo to generate a response based on context
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides clear and concise legal explanations based on provided documents."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_message}"}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        # Extract and send the AI-generated response
        answer = response.choices[0].message.content
        await cl.Message(content=answer).send()

    except Exception as e:
        logging.error(f"Error during message processing: {str(e)}")
        await cl.Message(content=f"🚨 Error: {str(e)}").send()
