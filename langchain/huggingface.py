import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
# loading env varables
from dotenv import load_dotenv
load_dotenv()


# load environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")




from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto", 
    # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)

print()


def generateText(text):
    response = chat_model.invoke(text).content
    return response
    

interface = gr.Interface(
    fn=generateText,
    inputs=["text"],
    outputs=["text"],
)

interface.launch()
    

