from langchain_openai import ChatOpenAI
import streamlit as st # type: ignore
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = ChatOpenAI(model='gpt-4o')

st.header("Research Tool: ")

paper_input = st.selectbox("Select Research Paper Name: ",
                           ["Attention is all you need",
                           "BERT: Pre training for bidirectional transformers",
                           "GPT-3: Language Models are few shot Learners",
                           "Diffusion Models Beat GANs on Image Synthesis."])

style_input = st.selectbox("Select Explanation style: ",
                           ["Beginner-Friendly",
                            "Technical",
                            "Code-Oriented",
                            "Mathematical"])

length_input = st.selectbox("Select Explanation Length: ",
                            ["Short(1-2 paragraphs) ",
                             "Medium(3-5 paragraphs)",
                             "Long(Detailed Explanation)"])

template = load_prompt('template.json')

prompt = template.format(
                    paper_input = paper_input,
                    style_input =style_input,
                    length_input = length_input)

# user_input = st.text_input("Enter prompt: ")

if(st.button('Summarise')):
    result = model.invoke(prompt)
    st.write(result.content)