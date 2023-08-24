import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import gradio as gr

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "r8_KtB7NRgsT3NQutRH3hvZtXGuPiKzIDQ1uix8J"

# Initialize Pinecone
pinecone.init(api_key='f5444e56-58db-42db-afd6-d4bd9b2cb40c', environment='asia-southeast1-gcp-free')

# Load and preprocess the PDF document
loader = PyPDFLoader('../rechat/2207.02696.pdf')
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Use HuggingFace embeddings for transforming text into numerical vectors
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# Set up the Pinecone vector database
index_name = "langchainpinecone"
index = pinecone.Index(index_name)
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)
# Initialize Replicate Llama2 Model
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 3000}
)

# Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)


chat_history = []
# Define the Gradio interface function
def chat_interface(prompt):
    result = qa_chain({'question': prompt, 'chat_history': chat_history})
    chat_history.append((prompt, result['answer']))
    
    conversation_history = "\n".join([f"You: {q}\nChatbot: {a}" for q, a in chat_history])
    
    return conversation_history

# Create the Gradio interface
iface = gr.Interface(
    fn=chat_interface,
    inputs="text",
    outputs="text",
    live=True,
    capture_session=True,
    title="Chatbot Interface",
    description="Ask questions and get answers from the chatbot."
)

# Run the Gradio interface
iface.launch()