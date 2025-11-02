from langchain_community.document_loaders import TextLoader
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

loader = TextLoader('7.Document_Loaders\cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs[0].metadata)

chain = prompt | model | parser
result = chain.invoke({'poem':docs[0].page_content})

print(result)