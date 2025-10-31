from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import Client
import datetime
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# 1️⃣ Define tools (with placeholder args)
@tool
def get_current_date(input_text: str = "") -> str:
    """Returns today's date."""
    return datetime.date.today().isoformat()

@tool
def add_numbers(input_str: str) -> float:
    """Adds two numbers given as 'a b'."""
    a, b = map(float, input_str.split())
    return a + b

# 2️⃣ Create Tool objects

# def get_current_date(_input=None):
#     """Returns today's date."""
#     return datetime.date.today().isoformat()

# def add_numbers(input_str):
#     """Adds two numbers given as 'a b'."""
#     a, b = map(float, input_str.split())
#     return a + b

# tools = [
#     Tool(
#         name="DateTool",
#         func=get_current_date,
#         description="Use this to get today's date."
#     ),
#     Tool(
#         name="AdditionTool",
#         func=add_numbers,
#         description="Use this to add two numbers, e.g., '3 5'."
#     )
# ]

tools = [get_current_date, add_numbers]

# 3️⃣ Load the latest ReAct prompt from LangChain Hub
client = Client()
prompt = client.pull_prompt("hwchase17/react")

# 4️⃣ Create the new style agent (no deprecation warning)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(llm, tools, prompt)

# 5️⃣ Create executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6️⃣ Run!
response = agent_executor.invoke({"input": "What is today's date?"})
print("AI:", response["output"])

response = agent_executor.invoke({"input": "Add 23 and 77"})
print("AI:", response["output"])
