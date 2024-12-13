from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_openai import ChatOpenAI
import gradio as gr
from dotenv import load_dotenv
import os
load_dotenv()

class ChatBot:
    def __init__(self):
        self.model_name = 'gemini-1.5-flash'
        self.load_model()
    
    def load_model(self,):
        self.llm = ChatGoogleGenerativeAI(model=self.model_name)

    def chat_bot(self,message,history):
        return self.llm.invoke(message).content
    
if __name__ == '__main__':
        #api_key = os.getenv("GOOGLE_API_KEY")
        #genai.configure(api_key=api_key)
        #for model in genai.list_models():
             #print(model.name)

    chatbot = ChatBot()
    gr.ChatInterface(chatbot.chat_bot).launch(share=True)

"""
models/chat-bison-001
models/text-bison-001
models/embedding-gecko-001
models/gemini-1.0-pro-latest
models/gemini-1.0-pro
models/gemini-pro
models/gemini-1.0-pro-001
models/gemini-1.0-pro-vision-latest
models/gemini-pro-vision
models/gemini-1.5-pro-latest
models/gemini-1.5-pro-001
models/gemini-1.5-pro-002
models/gemini-1.5-pro
models/gemini-1.5-pro-exp-0801
models/gemini-1.5-pro-exp-0827
models/gemini-1.5-flash-latest
models/gemini-1.5-flash-001
models/gemini-1.5-flash-001-tuning
models/gemini-1.5-flash
models/gemini-1.5-flash-exp-0827
models/gemini-1.5-flash-002
models/gemini-1.5-flash-8b
models/gemini-1.5-flash-8b-001
models/gemini-1.5-flash-8b-latest
models/gemini-1.5-flash-8b-exp-0827
models/gemini-1.5-flash-8b-exp-0924
models/gemini-2.0-flash-exp
models/gemini-exp-1206
models/gemini-exp-1121
models/gemini-exp-1114
models/learnlm-1.5-pro-experimental
models/embedding-001
models/text-embedding-004
models/aqa
"""