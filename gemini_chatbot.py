import google.generativeai as genai
from langchain_openai import ChatOpenAI
import gradio as gr
from dotenv import load_dotenv
import os
load_dotenv()

class ChatBot:
    def __init__(self):
        self.model_name = 'gemini-1.5-flash'
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.load_model()
    
    def load_model(self,):
        self.llm = genai.GenerativeModel(model_name=self.model_name)

    def chat_bot(self,message,history):
        print("History : ", history)
        return self.llm.generate_content(message).text 
    
if __name__ == '__main__':
    chatbot = ChatBot()
    gr.ChatInterface(chatbot.chat_bot).launch(share=True)