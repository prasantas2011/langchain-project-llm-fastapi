from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# 1️⃣ Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history")

# 2️⃣ Create a prompt that includes memory
prompt = PromptTemplate.from_template(
    """
    You are a friendly AI assistant.
    The conversation so far: {chat_history}
    Human: {human_input}
    AI:"""
)

# 3️⃣ Load your LLM
llm = OpenAI(
    model="gpt-4o-mini",  # uses OpenAI's instruct-style model
    temperature=0.7
)

# 4️⃣ Create a chain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# 5️⃣ Simulate a conversation
chain.invoke({"human_input": "Hi, who are you?"})
chain.invoke({"human_input": "Can you remind me what I asked earlier?"})
