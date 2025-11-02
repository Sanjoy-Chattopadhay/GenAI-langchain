from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
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

loader = DirectoryLoader(
    path='7.Document_Loaders\PPTS',
    glob = '*.pdf',
    loader_cls=PyPDFLoader
)

#load vs lazy_load()
docs = loader.lazy_load()

# print(len(docs))
# print(docs[0].metadata)

for document in docs:
    print(document.metadata)