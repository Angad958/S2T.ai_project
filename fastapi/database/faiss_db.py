from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
from hashlib import md5

def generate_unique_id(text):
    """Generate a unique ID for a document based on its content."""
    return md5(text.encode('utf-8')).hexdigest()

def initialize_vector_db():
    # Load the text file and split into documents
    loader = TextLoader("data.txt", encoding='utf-8')
    documents = loader.load()

    if not documents:
        print("No documents loaded. Ensure 'data.txt' exists and is not empty.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    if not docs:
        print("No chunks created. Check the input documents.")
        return

    print(f"Total chunks created: {len(docs)}")

    # Initialize embeddings using Ollama
    embeddings = OllamaEmbeddings(model="phi3", base_url='http://localhost:11434/')

    # Test embedding generation
    try:
        test_embedding = embeddings.embed_query("Test text for embedding generation.")
        print(f"Sample embedding length: {len(test_embedding)}")
    except Exception as e:
        print(f"Error generating sample embedding: {e}")
        return

    # Initialize FAISS vector store
    if len(docs) == 0:
        print("No documents to process. Exiting.")
        return

    try:
        index = FAISS.from_documents(docs, embeddings)
        index.save_local("faiss_index")
        print("FAISS index successfully created and saved locally.")
    except Exception as e:
        print(f"Error creating FAISS index: {e}")

if __name__ == "__main__":
    initialize_vector_db()
