from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large',
                             dimensions=300)

document = [

    "Virat Kohli is an Indian cricketer known for his aggressive batting style and consistency across all formats. He has served as the captain of the Indian team and is one of the highest run-scorers in international cricket.",
    
    "Sachin Tendulkar, often called the 'God of Cricket', is regarded as one of the greatest batsmen in the history of the sport. He holds numerous records, including 100 international centuries and over 34,000 runs.",
    
    "MS Dhoni, a calm and strategic leader, captained India to major ICC victories including the 2007 T20 World Cup, 2011 ODI World Cup, and 2013 Champions Trophy. He is known for his lightning-fast wicketkeeping and finishing skills.",
    
    "Ben Stokes is an English all-rounder famous for his explosive batting, sharp bowling, and incredible match-winning performances, such as the 2019 World Cup final and the Headingley Ashes Test.",
    
    "AB de Villiers, known as 'Mr. 360', is a South African cricketer admired for his innovative stroke play and ability to hit the ball to any part of the ground. He has redefined modern batting with his versatility and creativity."
]

query = 'tell me about stokes'

doc_embeddings = embedding.embed_documents(document)
query_embeddings = embedding.embed_query(query)

# print(cosine_similarity([query_embeddings], doc_embeddings))

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x : x[1])[-1]

print("Query: ", query)
print(document[index])
print("Similarity_Score : ", score)
