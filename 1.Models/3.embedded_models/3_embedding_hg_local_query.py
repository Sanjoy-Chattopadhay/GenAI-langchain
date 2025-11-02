from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding = HuggingFaceEmbeddings(model = 'sentence-transformers/all-MiniLM-L6-v2')


text = "Delhi is the capital of India"

vector = embedding.embed_query(text)

# Print vector length or shape
print(f"Vector length: {len(vector)}")
# print(vector[:10])  # show first 10 values