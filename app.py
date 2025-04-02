from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

## Env Variable
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")

print(os.environ.get("SSL_CERT_FILE"))

## Prompt Template
prompt = ChatPromptTemplate(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)

## Streamlit Framework
st.title("Chatbot with GROQ AI")
input_text = st.text_input("Search the topic you want")

## Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

## Definint the Type of Output Parser
output_parser = StrOutputParser()

## Create the chain that guarantees JSON output
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
    