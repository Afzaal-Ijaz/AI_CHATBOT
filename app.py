import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import openai
from dotenv import load_dotenv
import warnings
import tempfile

warnings.filterwarnings("ignore")

load_dotenv()

# openai.api_key = os.getenv("OPEN_AI_KEY")

st.set_page_config(page_title = "AI Chatbot", layout= "centered")
st.title("AI Chatbot")
st.subheader("Built with streamlit, langchain and GPT-4o")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "conversation" not in st.session_state:
    
    llm = ChatOpenAI(
        model_name = "gpt-4o",
        temperature = 0.7,
        openai_api_key = os.getenv("OPEN_AI_KEY")
    )
    
    memory = ConversationBufferMemory(return_messages = True)
    
    st.session_state.conversation = ConversationChain(
        llm = llm,
        memory = memory,
        verbose= False
    )
    
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
            
    else:
        with st.chat_message("assistant"):
            st.write(message.content)
    
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content= user_input))
    
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.predict(input = user_input)
            st.write(response)
            
    st.session_state.chat_history.append(AIMessage(content = response))
    
with st.sidebar:
    st.title("Option")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
    
        memory = ConversationBufferMemory(return_messages = True)
    
        llm = ChatOpenAI(
            model_name = "gpt-4o",
            temperature = 0.7,
            openai_api_key = os.getenv("OPEN_AI_KEY")
        )
     
        st.rerun() 
        
# st.subheader("About")

