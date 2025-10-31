from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define the LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

# Step 2: Define a reusable prompt template
#single variable example
# template = """
# You are an expert AI tutor.
# Write a short paragraph explaining {topic} in simple terms for beginners.
# """

template = """
Explain {concept} using a fun example involving {character}.
Keep it under 100 words.
"""

prompt = PromptTemplate(
    #input_variables=["topic"],  # defines placeholders singlely
    input_variables=["concept", "character"],
    template=template
)

# Step 3: Fill the template dynamically
#final_prompt = prompt.format(topic="Artificial Intelligence") # single variable
final_prompt = prompt.format(concept="neural networks", character="Spider-Man")

# Step 4: Send it to the model
response = llm.invoke(final_prompt)

print("Prompt:", final_prompt)
print("AI Response:", response)
