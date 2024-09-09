from transformers import pipeline  # Importing the text generation pipeline from Hugging Face
from langchain_core.prompts import ChatPromptTemplate  # Importing template for chat prompts
from langchain_core.output_parsers import StrOutputParser  # Importing parser to format the output as a string

import streamlit as st  # Importing Streamlit to build the web interface
import os  # Importing os for handling environment variables
from dotenv import load_dotenv  # Importing to load environment variables from a .env file

load_dotenv() 

model_name = "gpt2"  # Setting the model name to GPT-2
llm = pipeline('text-generation', model=model_name)  # Initializing the text generation pipeline with GPT-2

# Creating a chat prompt template with system and user messages
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to my queries"),  # System context message
        ("user", "Question:{question}")  # Placeholder for the user's question
    ]
)

st.title('Langchain Demo with GPT-2')  # Setting the title of the Streamlit app
input_text = st.text_input("Search the topic you want")  # Text input field for user queries

output_parser = StrOutputParser()  # Setting up the output parser

# Function to generate a response from the model
def generate_response(question):
    result = llm(question, max_length=50)  # Generating a response with the model, adjusting max_length as needed
    return result[0]['generated_text']  # Extracting and returning the generated text from the result

# If there's input text from the user
if input_text:
    response = generate_response(input_text)  # Generating a response based on the input text
    st.write(response)  # Displaying the generated response in the Streamlit app
