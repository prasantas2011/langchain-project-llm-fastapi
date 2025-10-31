# resume_local_rag.py
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Load resume ---
loader = PyPDFLoader("resume.pdf")
pages = loader.load()
resume_text = "\n".join([p.page_content for p in pages])

# --- Load a local model ---
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # or "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

print("✅ Resume loaded. Ask questions (type 'exit' to stop).")

while True:
    question = input("\nAsk: ")
    if question.lower() == "exit":
        break

    prompt = f"Resume:\n{resume_text}\n\nQuestion: {question}\nAnswer based only on the resume:"
    result = pipe(prompt)[0]['generated_text']
    print("\nAnswer:", result.split("Answer:")[-1].strip())
