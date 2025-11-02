from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.structured import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = StructuredOutputParser.fr

template = PromptTemplate(
    template = 'Give me all names, age and city of a fictional cartoon coco movie \n {format_instruction}',
    input_variables = [],
    partial_variables={'format_instruction' : parser.get_format_instructions}
)


schema = [
    ResponseSchema(name = 'fact1', description = 'Fact 1 about the topic'),
    ResponseSchema(name = 'fact2', description = 'Fact 2 about the topic'),
    ResponseSchema(name = 'fact3', description = 'Fact 3 about the topic')
]