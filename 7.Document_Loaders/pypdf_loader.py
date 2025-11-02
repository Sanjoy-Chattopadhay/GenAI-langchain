from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = OpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Generate a two point summary from this poem.{poem}',
    input_variables=['poem']
)

loader = PyPDFLoader('7.Document_Loaders\linear_algebra.pdf')

docs = loader.load()

print(len(docs))


# chain = prompt | model | parser
# result = chain.invoke({'poem':docs[0].page_content})

# print(result)