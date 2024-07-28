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
from flask import Flask, request, abort

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Flask App ---
app = Flask(__name__)

TELEGRAM_TOKEN = os.getenv("7268994371:AAEelLT9RlYb_jffqiUv-P6wRyi24rNTMws")
GOOGLE_API_KEY = os.getenv("AIzaSyCGZ-iE1paU93oSY5NHRWA_F8gf3Gs0sCg")
PERSIST_DIR = 'db/gemini/'  # Replace with your actual directory

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "HR bot"

# Initialize chat history
history = []

# Initialize the Gemini Pro 1.5 model
model = ChatGoogleGenerativeAI(
    model="gemini-pro", temperature=0.1, convert_system_message_to_human=True
)

# Configure Google Generative AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Data Pre-processing (Chroma) ---
if not os.path.exists(PERSIST_DIR):
    pdf_loader = DirectoryLoader("data/", glob="./*.pdf", loader_cls=PyPDFLoader)

    try:
        pdf_documents = pdf_loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        pdfs = splitter.split_text(pdf_context)
        vectordb = Chroma.from_texts(
            pdfs, embeddings, persist_directory=PERSIST_DIR
        )
        vectordb.persist()
    except Exception as e:
        logger.error(f"Error loading and processing PDF documents: {e}")
        raise
else:
    try:
        vectordb = Chroma(
            persist_directory=PERSIST_DIR, embedding_function=embeddings
        )
    except Exception as e:
        logger.error(f"Error loading persisted vector database: {e}")
        raise

# Initialize retriever and query chain
retriever = vectordb.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)
query_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

# --- Telegram Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hello! I am an HR chatbot for Acme AI. How can I assist you?"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    history.append({"role": "user", "content": user_message})

    prompt = (
        "You are an expert HR chatbot with access to company policies, guidelines, and other HR-related documents."
        "Provide detailed and specific answers to the following question based on the available documents:"
        "Your name is Acme AI HR bot."
        "Thoroughly read and utilize the HR manual to provide comprehensive and accurate responses to any question asked."
        "Summarize too large messages into a short detailed message."
        "When user asks or tell something in any other language than English, you will then reply in that specific language."
    )

    # Formulate the complete query
    query = f"{prompt}\n\nUser Question: {user_message}"

    # Get the response from the query chain
    response = query_chain({"query": query})
    bot_response = response["result"]

    history.append({"role": "assistant", "content": bot_response})

    # Check for empty response
    if not bot_response:
        bot_response = "I'm sorry, I couldn't find an answer to your question."

    # Split long messages
    if len(bot_response) > 4096:
        for i in range(0, len(bot_response), 4096):
            await update.message.reply_text(bot_response[i : i + 4096])
    else:
        await update.message.reply_text(bot_response)

# --- Telegram Bot Setup ---
bot = ApplicationBuilder().token('7268994371:AAEelLT9RlYb_jffqiUv-P6wRyi24rNTMws').build()
bot.add_handler(CommandHandler("start", start))
bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# --- Webhook Route ---
@app.route('/7268994371:AAEelLT9RlYb_jffqiUv-P6wRyi24rNTMws', methods=['POST'])
def webhook_handler():
    if request.method == 'POST':
        update = Update.de_json(request.get_json(force=True), bot.bot) 
        bot.process_update(update)
        return 'ok'  
    else:
        abort(403) 

# --- Run Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
