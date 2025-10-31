from langchain_community.document_loaders import TextLoader,PyPDFLoader
#from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# 1️⃣ Load Documents
# loader = TextLoader("docs/my_knowledge.txt")
# documents = loader.load()

#or

loader = PyPDFLoader("resume.pdf")
documents = loader.load()

# 2️⃣ Split into Chunks
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(documents)

# 3️⃣ Create Embeddings + Vector Store
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# 4️⃣ Create Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 5️⃣ Create LLM + RetrievalQA Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# 6️⃣ Ask Questions
#query = "What is LangChain and how does RAG help it?"
query = "What skill resume have ,please read all content and summerize the skill?"
result = qa_chain.invoke({"query": query})

print("\n🔍 Question:", query)
print("💬 Answer:", result["result"])
print("\n📚 Sources:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "Unknown"))
