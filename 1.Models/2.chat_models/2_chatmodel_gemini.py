from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load your .env file (contains GOOGLE_API_KEY)
load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0
)

# Ask a question
result = model.invoke("What is the capital of India?")

# Print response text
print(result.content)
