from fastapi import FastAPI, Query
from dotenv import load_dotenv

# LangChain loaders and modules (community versions)
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS



# Core utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

# Model integrations
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI(title="Multi-Source RAG Chatbot")

# --- 🧠 Initialize Model and Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# --- 🗂️ PDF Loader ---
pdf_loader = PyPDFLoader("resume.pdf")
pdf_docs = pdf_loader.load()

# --- 🌐 Website Loader ---
web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/India")
web_docs = web_loader.load()

# --- ✂️ Text Splitter ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_docs = pdf_docs + web_docs
chunks = splitter.split_documents(all_docs)

# --- 🧱 FAISS Vector Store ---
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# --- 🧾 Database Connection (optional) ---
try:
    #db = SQLDatabase.from_uri("mysql+mysqlconnector://root:@localhost/llmdb")
    db = None
    db_chain = None

except Exception as e:
    db = None
    db_chain = None
    print("⚠️ Database not connected:", e)

# --- RAG Chains ---
pdf_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
web_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Routing Logic ---
def route_question(question: str):
    q = question.lower()
    if "resume" in q or "skill" in q:
        return "pdf"
    elif "india" in q or "prime minister" in q:
        return "web"
    elif "employee" in q or "salary" in q:
        return "db"
    else:
        return "llm"

@app.get("/ask")
async def ask_question(question: str = Query(..., description="Ask any question")):
    try:
        source = route_question(question)
        print(f"🧭 Using source: {source.upper()}")

        if source == "pdf":
            result = pdf_qa.invoke({"query": question})
            answer = result.get("result", "No answer found.")
            src = ["resume.pdf"]

        elif source == "web":
            result = web_qa.invoke({"query": question})
            answer = result.get("result", "No answer found.")
            src = ["https://en.wikipedia.org/wiki/India"]

        elif source == "db" and db_chain:
            answer = db_chain.run(question)
            src = ["MySQL Database"]

        else:
            # fallback to direct LLM
            llm_answer = llm.invoke(question)
            answer = llm_answer.content if llm_answer else "No answer found."
            src = ["LLM (general knowledge)"]

        # Fallback if no confident answer
        if not answer or "i don't know" in answer.lower():
            return {"answer": "No answer found.", "sources": []}

        return {"answer": answer, "sources": src}

    except Exception as e:
        return {"answer": "No answer found.", "sources": [], "error": str(e)}
