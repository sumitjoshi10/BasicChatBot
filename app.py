from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

import streamlit as st
import os
from dotenv import load_dotenv

## Env Variable
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def groq_ai_chatbot():
    ## Prompt Template
    prompt = ChatPromptTemplate.from_messages(
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
        
def hugging_face_chatbot():
    ## Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","You are a helpful assistant. Please response to the user queries"),
            ("user", "Question:{question}")
        ]
    )

    ## Streamlit Framework
    st.title("Chatbot with HUGGING FACE")
    input_text = st.text_input("Search the topic you want")

    ## Initialize Groq LLM
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    ## Definint the Type of Output Parser
    output_parser = StrOutputParser()

    ## Create the chain that guarantees JSON output
    chain = prompt | llm | output_parser
    
    if input_text:
        st.write(chain.invoke({"question": input_text}))
    
def main():
    st.set_page_config(page_title="Basic Chat Bot")
    
    st.sidebar.title("Navigator")
    selection = st.sidebar.radio("Go to", ["Groq AI", "Hugging Face"])
    
    if selection == "Groq AI":
        groq_ai_chatbot()
    elif selection == "Hugging Face":
        hugging_face_chatbot()
    
    st.markdown("---")
    st.markdown("Sumit Joshi")
    
if __name__ == "__main__":
    main()
    