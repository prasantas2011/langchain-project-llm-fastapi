from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# 1️⃣ Define a prompt template
template = "Explain {topic} in a simple way a 10-year-old could understand."
prompt = PromptTemplate.from_template(template)

# 2️⃣ Load LLM
llm = OpenAI(
    model="gpt-4o-mini",  # uses OpenAI's instruct-style model
    temperature=0.7
)

# 3️⃣ Create a chain
chain = LLMChain(prompt=prompt, llm=llm)

# 4️⃣ Run the chain
result = chain.invoke({"topic": "Quantum Computing"})
print(result)
