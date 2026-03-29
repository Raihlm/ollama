import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever

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

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

# rag prompt
template = """You are a helpful assistant for answering questions based on the retrieved information from the document.
Use only the following retrieved information to answer the question. If you don't know the answer, say you don't know.
{context}"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | StrOutputParser()
)

res = chain.invoke("What are the main findings of the paper?")

print(res)