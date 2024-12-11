import os
from fastapi import HTTPException
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

general_system_template = """ 
You are a cricket enthusiast who specializes in the FAISS vector database. Your task is to process user queries about cricketers using the FAISS vector database. Only respond when the query is relevant to the context of vector databases or their applications.
- Provide responses in complete sentences, addressing the user.
- Restrict your answers strictly to the context of the vector database. For example, mention how vector search might help find cricketers with specific skills or attributes.
- Return an empty response if the user query does not align with the context of the vector database.
----
{context}
----

"""

general_user_template = "Question: {question}"


def load_vector_db(model_name: str = "phi3", base_url: str = "http://ollama:11434/"):
    """
    Load the FAISS vector database from the local faiss_index directory.
    """
    try:
        # Absolute path to the FAISS database
        base_dir = os.path.dirname(os.path.abspath(__file__))
        faiss_path = os.path.join(base_dir, "../database/faiss_index")  # Constructing absolute path

        # Loading embeddings and FAISS index
        embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
        return db

    except Exception as e:
        print(f"Error loading FAISS vector DB: {e}")
        raise HTTPException(status_code=500, detail="Failed to load FAISS vector DB")


def create_rag_chain(vectorstore) -> ConversationalRetrievalChain:

   # Creating a ConversationalRetrievalChain that uses the FAISS vectorstore as a retriever and
   # Ollama (phi3) as the LLM for RAG, with custom prompts.
   
    try:
        # Define the prompt templates
        messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        llm = OllamaLLM(model="phi3", base_url="http://ollama:11434/")
        retriever = vectorstore.as_retriever()
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={'prompt': qa_prompt}
        )
        print("RAG chain created successfully.")
        return rag_chain

    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        raise HTTPException(status_code=500, detail="Failed to create RAG chain")


def search_DB(query: str, chat_history=None) -> str:
    """
    Search the FAISS vector database with a user query.
    Supports conversational context with chat_history.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Missing query parameter")

    if chat_history is None:
        chat_history = []  # Initializing chat_history as an empty list for the first query

    try:
        print("Loading FAISS vector database...")
        vectorstore = load_vector_db()
        print("Creating RAG chain...")
        rag_chain = create_rag_chain(vectorstore)
        print("Invoking RAG chain...")

        # Run the query through the chain
        result = rag_chain.invoke({"question": query, "chat_history": chat_history})
        print(f"Query result: {result}")
        return result["answer"]

    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch opportunities from LLM")

