from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a detail;ed report about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate 2 pointer summary from this given text: {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model| parser 

result = chain.invoke({'topic':'cricket'})

print(result)

# chain.get_graph().print_ascii()