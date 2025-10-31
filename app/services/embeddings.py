from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class HFEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    # LangChain expects these two methods
    def embed_documents(self, texts):
        """Embed a list of documents."""
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        """Embed a single query."""
        return self.model.encode([text], convert_to_tensor=False)[0]
