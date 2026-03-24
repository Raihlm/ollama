import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

ollama.pull("nomic-embed-text")

doc_path = "../file/indoMentalHealthPaper.pdf"
multiModalModel = "llava"
standardModel = "llama3.2"

if doc_path:
    Loader = UnstructuredPDFLoader(doc_path)
    documents = Loader.load()
    print('success', documents)
else:
    print(" No document path provided.")

content = documents[0].page_content
print(content[:1000])

# split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(documents)
#print(f"Number of chunks: {len(chunks)}")

# store to vector db
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="papersRAG"
)


# Retrieval
llm = ChatOllama(model=standardModel)


QUERY_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="Given the following query, retrieve relevant information from the document: {query}"
)