from transformers import pipeline  # Importing the text generation pipeline from Hugging Face
from langchain_core.prompts import ChatPromptTemplate  # Importing the template for chat prompts
from langchain_core.output_parsers import StrOutputParser  # Importing parser to format the output as a string
from huggingface_hub import login  # Importing function to log in to Hugging Face
import streamlit as st  # Importing Streamlit for building the web interface
import os  # Importing os for environment variable handling
from dotenv import load_dotenv  # Importing to load environment variables from a .env file

load_dotenv()  
hf_token = "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"  # Hugging Face token for model access
login(hf_token)  # Logging in to Hugging Face to authenticate access to gated models

# Setting up the text generation pipeline with the Llama-2 model
llama_pipeline = pipeline("text-generation",
                         model="meta-llama/Llama-2-7b-hf",  
                         use_auth_token=hf_token)  # Using the token to access the gated model

# Creating a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to my queries"),  # System message for context
        ("user", "Question:{question}")  # Placeholder for user’s question
    ]
)

st.title('Langchain Demo with llama')  # Setting the title of the Streamlit app
input_text = st.text_input("Say something")  # Text input field for user queries

# Setting up the output parser
output_parser = StrOutputParser()

# If there's input text from the user
if input_text:
    response = llama_pipeline(input_text, max_length=200, do_sample=True)  # Generating a response from the model
    st.write(response[0]['generated_text'])  # Displaying the generated response in the Streamlit app
