from fastapi import FastAPI, Query
from dotenv import load_dotenv

# --- LangChain imports ---
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

load_dotenv()

app = FastAPI(title="Multi-Source RAG Chatbot")

# --- 🧠 Initialize Model and Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 🗂️ Load Data Sources ---
pdf_loader = PyPDFLoader("resume.pdf")
pdf_docs = pdf_loader.load()

web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/India")
web_docs = web_loader.load()

# --- ✂️ Text Splitter ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_docs = pdf_docs + web_docs
chunks = splitter.split_documents(all_docs)

# --- 🧱 Create FAISS Vector Store ---
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# --- 🧾 Database Connection ---
try:
    db = SQLDatabase.from_uri("sqlite:///mydb.sqlite")
    db_chain = create_sql_query_chain(llm, db)
    print("✅ Database connected successfully.")
except Exception as e:
    db = None
    db_chain = None
    print("⚠️ Database not connected:", e)

# --- RAG Chains ---
pdf_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
web_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- 🔍 Route question based on content ---
def route_question(question: str):
    q = question.lower()
    if "resume" in q or "skill" in q:
        return "pdf"
    elif "india" in q or "prime minister" in q:
        return "web"
    elif "user" in q or "salary" in q or "employee" in q:
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
            # Step 1: Ask LLM to generate SQL
            sql_response = db_chain.invoke({"question": question})
            print("🧠 Raw LLM SQL Output:", sql_response)

            # Step 2: Extract the SQL statement only
            if isinstance(sql_response, dict):
                query_text = sql_response.get("result", "")
            else:
                query_text = str(sql_response)

            # Remove LangChain’s "SQLQuery:" prefix if present
            query_text = query_text.replace("SQLQuery:", "").strip()
            print(f"🧩 Final SQL to execute: {query_text}")

            # Step 3: Run the SQL on SQLite
            db_result = db.run(query_text)

            # Step 4: Format result nicely
            if isinstance(db_result, list):
                formatted = [dict(zip(["column_" + str(i) for i in range(len(row))], row)) for row in db_result]
            else:
                formatted = str(db_result)

            answer = formatted
            src = ["SQLite Database"]
        else:
            # fallback to direct LLM response
            llm_answer = llm.invoke(question)
            answer = llm_answer.content if llm_answer else "No answer found."
            src = ["LLM (general knowledge)"]

        return {"answer": answer, "source": source, "sources": src}

    except Exception as e:
        return {"answer": "No answer found.", "error": str(e), "sources": []}
