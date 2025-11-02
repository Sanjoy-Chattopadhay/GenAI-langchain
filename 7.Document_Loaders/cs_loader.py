from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize model and parser
model = OpenAI()
parser = StrOutputParser()

loader = CSVLoader('7.Document_Loaders\Social_Network_Ads.csv')

# Load webpage content
docs = loader.load()

# Print some sample output
print(docs[0].metadata)
print(docs[0].page_content[:1000])  # show first 1000 chars

