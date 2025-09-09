# now we have to define simple graph

import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# from IPython.display import Image,display
from typing import TypedDict, Literal
import random

# from langchain_huggingface import HuggingFaceEmbeddings


# loading env varables
from dotenv import load_dotenv

load_dotenv()


# load environment variables
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# llm model
# llm = ChatGroq(model="llama-3.1-8b-instant")


# state
class State(TypedDict):
    graph_state: str


def node_1(state: State):
    # this fucntion are override the default value of state
    print("+++++node_one++++")
    return {"graph_state": state["graph_state"] + " " + "node1"}


def node_2(state: State):
    print("+++++node_two++++")
    return {"graph_state": state["graph_state"] + " " + "node2"}


def node_3(state: State):
    print("+++++node_three++++")
    return {"graph_state": state["graph_state"] + " " + "node3"}


# conduction edge


def decide_mood(state: State) -> Literal["node_2", "node_3"]:
    if random.random() < 0.5:
        return "node_2"
    return "node_3"


# building graph


builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)


# add
graph = builder.compile()

print(graph.invoke({"graph_state": "graph state"}))

with open("diagram.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("Saved diagram.png")
