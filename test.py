import os
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Replicate API token
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")

# Pinecone API key
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment='asia-southeast1-gcp-free')

# Initialize loader
pdf_path = None
loader = None

# Use HuggingFace embeddings for transforming text into numerical vectors
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Set up the Pinecone vector database
index_name = "langchainpinecone"
index = pinecone.Index(index_name)
vectordb = None

# Initialize Replicate Llama2 Model
llm = None

def initialize_llm():
    global llm
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={"temperature": 0.75, "max_length": 3000}
    )

def initialize_qa_chain():
    global qa_chain, vectordb, llm
    if vectordb and llm:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vectordb.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True
        )

initialize_llm()  # Initialize the llm object
initialize_qa_chain()  # Initialize the qa_chain
chat_history = []

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        if user_input:
            result = qa_chain({'question': user_input, 'chat_history': chat_history})
            chat_history.append(("You", user_input))
            chat_history.append(("Chatbot", result['answer']))
    
    return render_template('index.html', chat_history=chat_history, show_chat=bool(vectordb))

@app.route('/pdf/<filename>')
def serve_pdf(filename):
    return send_from_directory('static', filename)

@app.route('/update_pdf', methods=['POST'])
def update_pdf():
    global pdf_path, loader, vectordb
    pdf_path = request.form['pdf_path']
    
    if pdf_path:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)
        initialize_qa_chain()  # Re-initialize the qa_chain
    
    return redirect(url_for('chat'))

if __name__ == '__main__':
    app.run(debug=True)
