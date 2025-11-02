from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set a default User-Agent to avoid warning
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# Initialize model and parser
model = OpenAI()
parser = StrOutputParser()

# Target URL
url = 'https://ibef.org/blogs/culinary-diplomacy-how-indian-food-festivals-boost-tourism-and-trade'
loader = WebBaseLoader(url)

# Load webpage content
docs = loader.load()

# Print some sample output
print(docs[0].metadata)
print(docs[0].page_content[:1000])  # show first 1000 chars

# Optional: summarize the content
prompt = PromptTemplate(
    template='Summarize this webpage in 2 concise points:\n\n{content}',
    input_variables=['content']
)

chain = prompt | model | parser
summary = chain.invoke({"content": docs[0].page_content})

print("\n=== SUMMARY ===\n", summary)
