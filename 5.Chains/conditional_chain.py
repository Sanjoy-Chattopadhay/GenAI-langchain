from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description = 'Give the sentiment of the feedback')


parser2 = PydanticOutputParser(pydantic_object=Feedback)


model1 = ChatOpenAI()

model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0
)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of this follwing feedback into positive or negative \n{feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions' : parser2.get_format_instructions()}
)

classfifier_chain = prompt1 | model1 | parser2


prompt2 = PromptTemplate(
    template = 'write an appropriate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', prompt2 | model1 | parser1),
    (lambda x : x.sentiment == 'negative', prompt3 | model1 | parser1),
    RunnableLambda(lambda x : "Count not find sentiment.")
)

chain = classfifier_chain | branch_chain

result = chain.invoke({'feedback' : 'this is a wonderful phone'})


print(result)


