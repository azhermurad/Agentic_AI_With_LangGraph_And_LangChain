from pprint import pprint
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState, StateGraph, START, END


# loading env varables
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")





def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools([multiply])


# define the state
class MessageState(MessagesState):
    pass


builder = StateGraph(MessagesState)

# adding nodes 
# node are simple python function which are used to update the state 

def chatbot(state:MessagesState):
    return {"messages":[]}




class State()