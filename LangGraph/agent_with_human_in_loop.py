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
        content="your are a helping assistant which help to answer the user query, you have to handle simple query and tool call as well and the answer of the tool should be in nice formate to user"
    )
]


# this will run after the model is run 
def chatbot(state: State):
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
config = {"configurable": {"thread_id": "2"}}

# graph_with_memory = builder.compile(checkpointer=memory)

graph = builder.compile(checkpointer=memory, interrupt_before=["chatbot"])

events = graph.stream(
    {"messages": [HumanMessage(content="add 2 and 2")]},
    config,
    stream_mode="values",
)


for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


print(graph.get_state(config).next)

# update the graph
graph.update_state(
    config, {"messages": [HumanMessage(content="no,actually add 5 and 5")]}
)


new_state = graph.get_state(config).values
for m in new_state["messages"]:
    m.pretty_print()

# now processed with our agent

events = graph.stream(
    None,  # this will start the processing where we have left s
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


events = graph.stream(
    None,  # this will start the processing where we have left s
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# is_approved = input("do you want to continue? yes/no")
# # this function will run when the user enter the yes in the input field
# if is_approved:
#     events = graph.stream(
#         None,  # this will start the processing where we have left s
#         config,
#         stream_mode="values",
#     )
#     for event in events:
#         if "messages" in event:
#             event["messages"][-1].pretty_print()
