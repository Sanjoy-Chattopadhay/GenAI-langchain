from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize LLM
llm = OpenAI(model="gpt-3.5-turbo-instruct")

# Run a simple query
result = llm.invoke("What is the capital of India?")
print(result)
