from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding = HuggingFaceEmbeddings(model = 'sentence-transformers/all-MiniLM-L6-v2')


documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

# Print vector length or shape
print(f"Vector length: {len(vector)}")
print(vector[:10])  # show first 10 values