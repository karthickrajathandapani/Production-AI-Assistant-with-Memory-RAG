from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama 
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# Load Documents
loader = PyPDFLoader("karthick resume.pdf")

documents = loader.load()

# split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

docs = splitter.split_documents(documents)

# Embeddings
embeddings = OllamaEmbeddings(
    model = "llama3.2"
)

# Vector DB
db = FAISS.from_documents(
    docs,
    embeddings
)

retriever = db.as_retriever()

# LLM
llm = Ollama(model="llama3.2")

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = retriever,
    memory = memory
)

print("Advanced AI Assistant Started")

while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        break
    
    result = qa_chain(
        {"question": query}
    )
    
    print("Bot:", result["answer"])