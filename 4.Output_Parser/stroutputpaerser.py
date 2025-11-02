from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI()

template1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a 5-line summary of the following text:\n\n{text}",
    input_variables=['text']
)
# With out stroutput parser

# prompt1 = template1.invoke({'topic': 'Black Hole'})
# prompt1_output = model.invoke(prompt1)
# text = prompt1_output.content

# prompt2 = template2.invoke({'text': text})
# prompt2_output = model.invoke(prompt2)

# print("\n=== SUMMARY ===")
# print(prompt2_output.content)


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'Black Hole'})
print(result) 