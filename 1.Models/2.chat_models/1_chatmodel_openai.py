from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load your API key
load_dotenv()

# Initialize Chat model
llm = ChatOpenAI(model="gpt-4o", temperature = 1.5)  # Use "gpt-4o" (the correct ID for GPT-4 Omni)

# Invoke the model
result = llm.invoke("suggest me 5 indian male names")

print(result.content)  # <-- use .content to print the text reply
