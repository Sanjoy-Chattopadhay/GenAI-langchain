from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me all names, age and city of a fictional cartoon coco movie \n {format_instruction}',
    input_variables = [],
    partial_variables={'format_instruction' : parser.get_format_instructions}
)

chain = template | model | parser

result = chain.invoke({})

print(result)