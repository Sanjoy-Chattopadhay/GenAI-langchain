from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

# Initialize model (you can specify model name)
model = ChatOpenAI(model="gpt-4o")

# Define output schema
class Review(TypedDict):
    key_themes : Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str,"a breif summary of the review"]
    sentiment: Annotated[str,"return the sentiment captured in the review either in negative, positive or neutral"]
    pros:Annotated[Optional[list[str]], 'write down all the pros inside a list']
    cons:Annotated[Optional[list[str]], "write down all the cons inside a list"]

# Enable structured output
structured_model = model.with_structured_output(Review)

review_text = """
I recently purchased a wooden queen-sized bed from Flipkart, and overall, I’m quite satisfied with the product. 
The delivery was prompt, and the packaging was sturdy, ensuring that all parts arrived safely. 
The assembly process was straightforward, with clear instructions provided, and the bed feels strong and well-built once set up. 
The finish gives it a premium look, and it complements my bedroom decor beautifully. 
The only minor downside is that the mattress support slats could have been slightly thicker for added durability. 
However, considering the price point and quality, it’s a great value-for-money purchase, and I’d definitely recommend it to anyone looking for a stylish and comfortable bed online.
"""

# Invoke structured model
result = structured_model.invoke(review_text)

for keys in result:
    print(keys, result[keys])
