import streamlit as st
import os
import logging
import ollama
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
""" from langchain_community.vectorstores import Chroma || 
ERROR OCCURED:
Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 
and will be removed in 1.0. 
An updated version of the class exists in the `langchain-chroma package and should be used instead. 
To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma`
vector_db.persist()""" 
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma


logging.basicConfig(level=logging.INFO)


# constants
DOC_PATH = "RAGproject/file/indoMentalHealthPaper.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "RAG_for_papers"
PERSIST_DIR = "./chroma_db"


def load_document(doc_path):
    """Load the document using appropriate loader based on file type."""
    if os.path.exists(doc_path):
        if doc_path.endswith('.pdf'):
            loader = PyPDFLoader(doc_path)
        else:
            loader = UnstructuredFileLoader(doc_path)
        documents = loader.load()
        logging.info(f"Loaded document with {len(documents)} pages.")
        return documents
    else:
        logging.error(f"Document path {doc_path} does not exist.")
        st.error("Document not found. Please check the path and try again.")
        return None
    

def split_document(documents):
    """Split the document into chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 300)
    chunks = text_splitter.split_documents(documents=documents)
    logging.info(f"Split document into {len(chunks)} chunks.")
    return chunks

@st.cache_resource
def load_vector_database():
    """load or create database"""

    # pull emedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIR):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIR
        )
        logging.info("Loaded existing vector database")
    else:
        #load & process document
        data = load_document(DOC_PATH)
        if data is None:
            return None
        
        chunks = split_document(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIR,
        )

        logging.info("Vector database created and persisted.")

    return vector_db

def create_retriever(vector_db, llm):
    """create retriever using multiquery retriever"""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
           different versions of the given user question to retrieve relevant documents from
           a vector database. By generating multiple perspectives on the user question, your
           goal is to help the user overcome some of the limitations of the distance-based
           similarity search. Provide these alternative questions separated by newlines.
           Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    logging.info("MultiQueryRetriever created successfully.")
    return retriever

def create_chain(retriever, llm):
    """create chain with preserved syntax for better readability"""
    template = """You are an AI assistant for question-answering tasks. Use the following retrieved documents to answer the question. If you don't know the answer, say you don't know. Always use all available information from the retrieved documents to provide a comprehensive answer.
    context:{context}
    Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm 
        |StrOutputParser()
    )

    return chain


def main():
    st.title("RAG Application with Ollama and Langchain")

    user_input = st.text_input("Enter your question about the document:")

    if user_input:
        with st.spinner("Processing your question...."):
            try:
                llm = ChatOllama(model=MODEL_NAME)

                vector_db = load_vector_database()

                if vector_db is None:
                    st.error("Failed to load or create vector database. Please check the logs for details.")
                    return
                
                retriever = create_retriever(vector_db, llm)

                chain = create_chain(retriever, llm)

                response = chain.invoke({"question": user_input})

                st.markdown("**Assistant:**")
                st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logging.error(f"Error processing user question: {e}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()