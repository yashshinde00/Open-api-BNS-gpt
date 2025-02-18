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
        similar_docs = similarity_search(user_message, k=3)

        if not similar_docs:
            await cl.Message(content="😕 No relevant documents found. Try another query.").send()
            return

        # Prepare the context from similar documents
        response_text = "\n\n".join([doc.page_content for doc in similar_docs])

        # Ensure the response does not exceed 2000 characters
        response_text = response_text[:2000]

        await cl.Message(content=response_text).send()

    except Exception as e:
        logging.error(f"Error during message processing: {str(e)}")
        await cl.Message(content=f"🚨 Error: {str(e)}").send()
