from pprint import pprint
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph, START, END


# loading env varables
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


def add(a: int, b: int) -> int:
    """add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> int:
    """divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools([multiply, add, divide])


# define the state
class State(MessagesState):
    pass


# node


# define the system message

system_message = [
    SystemMessage(
        content="your are a helping assistant which help to answer the user query, you have to handle simple query and tool call as well"
    )
]


def chatbot(state: State):
    print(state["messages"])
    return {"messages": [llm_with_tools.invoke(system_message + state["messages"])]}


# create the graph state

builder = StateGraph(State)


# add node in the graph (node are used to update the state of the graph)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools=[multiply, add, divide]))

# add edges in the graph
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")







# add memory to the list

from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

graph_with_memory = builder.compile(checkpointer=memory)

answers = graph_with_memory.invoke(
    {"messages": [HumanMessage(content="Hi there! My name is Azher Ali.")]}, config
)

for m in answers["messages"]:
    m.pretty_print()


user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
answers = graph_with_memory.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    config
    
)

for m in answers["messages"]:
    m.pretty_print()


snapshot = graph_with_memory.get_state(config)
print(snapshot)

# act let the model to call the specfic tool from the llm
# observe : pass the tool output back to the model
# reason: let the model reason about the tool output to decides what to do next(or called another tool or direct the output to the user)
