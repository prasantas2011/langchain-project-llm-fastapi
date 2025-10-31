# resume_rag.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- Set your OpenAI key (you can also export it in environment) ---
load_dotenv()

# --- Load your resume PDF ---
loader = PyPDFLoader("resume.pdf")
pages = loader.load()

# --- Combine all text from pages ---
full_text = "\n".join([page.page_content for page in pages])

# --- Initialize the LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Interactive Q&A loop ---
print("✅ Resume loaded. You can now ask any question about it!")
print("Type 'exit' to stop.\n")

while True:
    question = input("Ask a question about your resume: ")
    if question.lower() == "exit":
        break

    prompt = f"""
You are an assistant analyzing a resume.

Resume content:
\"\"\"{full_text}\"\"\"

Question: {question}

Answer clearly based only on the above resume.
"""
    response = llm.invoke(prompt)
    print("\nAnswer:", response.content)
