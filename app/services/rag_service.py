import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from .embeddings import HFEmbeddings
from ..utils.pdf_reader import extract_text_from_pdf

# 🧠 Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)

class RAGService:
    def __init__(self, index_dir="faiss_index", model_name="all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.model_name = model_name
        os.makedirs(index_dir, exist_ok=True)
        load_dotenv()
        self.embeddings = HFEmbeddings(model_name=model_name)

    def create_docs(self, text):
        logging.info("📄 Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_text(text)
        logging.info(f"✅ Created {len(chunks)} chunks.")
        docs = [Document(page_content=chunk) for chunk in chunks]
        return docs

    def build_index(self, pdf_path):
        logging.info(f"📚 Building FAISS index from PDF: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        logging.info(f"✅ Extracted {len(text)} characters from PDF.")
        
        docs = self.create_docs(text)
        logging.info("⚙️ Generating embeddings and building FAISS index...")
        
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        vectorstore.save_local(self.index_dir)
        logging.info(f"✅ Index created and saved at: {self.index_dir}")
        return "Index created successfully."

    def load_index(self):
        logging.info(f"📦 Loading FAISS index from: {self.index_dir}")
        return FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)

    def ask_question(self, query, similarity_threshold=0.7):
        logging.info(f"🧠 Received query: {query}")
        vectorstore = self.load_index()

        # Manually get relevant documents with similarity scores
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=4)

        # If no relevant document above threshold → no answer
        if not docs_and_scores or all(score < similarity_threshold for _, score in docs_and_scores):
            return {
                "answer": "No answer found.",
                "sources": []
            }
        
        # Extract relevant docs
        relevant_docs = [doc for doc, score in docs_and_scores if score >= similarity_threshold]
        print(relevant_docs)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        logging.info("🔍 Retrieving top 4 matching chunks...")

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        logging.info("🤖 Querying LLM for answer...")
        response = qa_chain.invoke({"query": query})

        logging.info("✅ LLM response generated successfully.")
        answer_text = response.get("result", "No answer found.")
        sources = [doc.page_content for doc in response.get("source_documents", [])]

        return {"answer": answer_text, "sources": sources}
        

    def ask_question_with_fallback(self, query, similarity_threshold=0.8):
        """
        Ask a question from the indexed PDF data using FAISS + GPT model.
        Returns consistent JSON: {"answer": "...", "sources": [...]}
        """
        try:
            # Load FAISS index
            vectorstore = self.load_index()

            # Retrieve top chunks with scores
            results = vectorstore.similarity_search_with_score(query, k=4)

            if not results:
                return {"answer": "No answer found.", "sources": []}

            # ✅ Smaller distance = more similar
            filtered_docs = [doc for doc, score in results if score < (1 - similarity_threshold)]

            # If too few results, keep the top one anyway
            if not filtered_docs:
                filtered_docs = [doc for doc, score in results[:1]]

            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # Initialize GPT model
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0
            )

            # Create RAG chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            # Run chain
            result = qa_chain.invoke({"query": query})

            # Extract answer & sources
            answer = result.get("result", "").strip()
            sources = [doc.page_content[:300] for doc in result.get("source_documents", [])]

            # ✅ Handle "I don't know" or empty answers
            if not answer or "i don't know" in answer.lower() or not sources:
                return {"answer": "No answer found.", "sources": []}

            return {"answer": answer, "sources": sources}

        except Exception as e:
            return {"answer": "No answer found.", "sources": [], "error": str(e)}


    def ask_question_with_fallback_new(self, query, similarity_threshold=0.8):
        """
        Ask a question from the indexed PDF data using FAISS + GPT model + fallback to LLM.
        Returns consistent JSON:
        {
            "answer": "...",
            "sources": [...]
        }
        """
        try:
            # ---- STEP 1: Load FAISS index ----
            vectorstore = self.load_index()
            results = vectorstore.similarity_search_with_score(query, k=4)

            # ---- STEP 2: Filter relevant docs ----
            docs_from_pdf = []
            if results:
                docs_from_pdf = [doc for doc, score in results if score < (1 - similarity_threshold)]
                if not docs_from_pdf:
                    docs_from_pdf = [doc for doc, score in results[:1]]  # keep at least top one

            # ---- STEP 3: Retriever and LLM setup ----
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            # ---- STEP 4: Create RAG chain ----
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            # ---- STEP 5: Try RAG (PDF) ----
            result = qa_chain.invoke({"query": query})
            answer = result.get("result", "").strip()
            sources = [doc.page_content[:300] for doc in result.get("source_documents", [])]

            # ---- STEP 6: If no answer or irrelevant, fallback to LLM ----
            if not answer or "i don't know" in answer.lower() or not sources:
                llm_direct_response = llm.invoke(f"Answer this question based on your general knowledge: {query}")

                # Extract text content properly
                llm_answer = (
                    llm_direct_response.content.strip()
                    if hasattr(llm_direct_response, "content")
                    else str(llm_direct_response).strip()
                )

                if llm_answer and "i don't know" not in llm_answer.lower():
                    return {
                        "answer": llm_answer,
                        "sources": ["LLM (general knowledge)"]
                    }

                # ---- STEP 7: Final fallback ----
                return {"answer": "No answer found.", "sources": []}

            # ---- STEP 8: Return PDF-based answer ----
            return {"answer": answer, "sources": sources}

        except Exception as e:
            return {"answer": "No answer found.", "sources": [], "error": str(e)}

