from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

llm = OpenAI(
    model="gpt-4o-mini",  # uses OpenAI's instruct-style model
    temperature=0.7
)

response = llm.invoke("Write a short haiku about LangChain.")
print("AI:", response)
