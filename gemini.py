import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import logging
import time

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("7268994371:AAEelLT9RlYb_jffqiUv-P6wRyi24rNTMws")
GOOGLE_API_KEY = os.getenv("AIzaSyCGZ-iE1paU93oSY5NHRWA_F8gf3Gs0sCg")
PERSIST_DIR = 'https://github.com/joti-acmeai/HR-bot/tree/main/db/gemini'  # Replace with your actual directory

# Initialize chat history
history = []

# Initialize the Gemini Pro 1.5 model
model = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    temperature=0.1, 
    convert_system_message_to_human=True
)

# Configure Google Generative AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if not os.path.exists(PERSIST_DIR):
    # Data Pre-processing
    pdf_loader = DirectoryLoader("https://github.com/joti-acmeai/HR-bot/tree/main/data", glob="./*.pdf", loader_cls=PyPDFLoader)
    
    try:
        pdf_documents = pdf_loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        pdfs = splitter.split_text(pdf_context)
        vectordb = Chroma.from_texts(pdfs, embeddings, persist_directory=PERSIST_DIR)
        vectordb.persist()
    except Exception as e:
        logger.error(f"Error loading and processing PDF documents: {e}")
        raise
else:
    try:
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    except Exception as e:
        logger.error(f"Error loading persisted vector database: {e}")
        raise

# Initialize retriever and query chain
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
query_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! I am an HR chatbot for Acme AI. How can I assist you?')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    history.append({'role': 'user', 'content': user_message})
    
    prompt = (
        "You are an expert HR chatbot with access to company policies, guidelines, and other HR-related documents."
        "Provide detailed and specific answers to the following question based on the available documents:"
        "Your name is Acme AI HR bot."
    )
    
    # Formulate the complete query
    query = f"{prompt}\n\nUser Question: {user_message}"
    
    # Get the response from the query chain
    response = query_chain({"query": query})
    bot_response = response['result']

    history.append({'role': 'assistant', 'content': bot_response})
    
    await update.message.reply_text(bot_response)

def main() -> None:
    retry_attempts = 5
    for attempt in range(retry_attempts):
        try:
            app = ApplicationBuilder().token('7268994371:AAEelLT9RlYb_jffqiUv-P6wRyi24rNTMws').build()
            
            app.add_handler(CommandHandler("start", start))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            
            logger.info("Bot is starting...")
            app.run_polling()
            break
        except telegram.error.NetworkError as e:
            logger.error(f"Network error: {e}. Attempt {attempt + 1} of {retry_attempts}")
            time.sleep(5)  # wait for 5 seconds before retrying
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

if __name__ == '__main__':
    main()
