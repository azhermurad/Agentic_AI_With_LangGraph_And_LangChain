from pprint import pprint
import os
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    RemoveMessage,
)
from typing import TypedDict, Annotated
from langgraph.graph import MessagesState, StateGraph, START, END


# loading env varables
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")


# define the state
class State(MessagesState):
    pass


# node
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def filterMessages(state: State):
    print(state,"dd")
    messages = state["messages"]
    return {"messages": [RemoveMessage(id=m.id) for m in messages[:-2]]}


# create the graph state
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("filter", filterMessages)


builder.add_edge(START, "filter")
builder.add_edge("filter", "chatbot")
builder.add_edge("chatbot", END)


graph = builder.compile()

with open("diagram.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("Saved diagram.png")




messages = [
    HumanMessage(content="Hello, how are you?"),
    AIMessage(content="I'm doing well, thank you for asking."),
    HumanMessage(content="Can you tell me a joke?"),
    AIMessage(content="Sure! Why don't scientists trust atoms? Because they make up everything!"),
    HumanMessage(content="Haha, that's a good one. Can you give me some advice on staying productive?"),
    AIMessage(content="Of course! Break your tasks into smaller chunks, use a timer like the Pomodoro technique, and take regular breaks."),
    HumanMessage(content="That sounds helpful. What about when I feel unmotivated?"),
    AIMessage(content="Try starting with a very small task to build momentum. Also, reminding yourself of the bigger picture can reignite motivation."),
    HumanMessage(content="Great tips! Can you recommend a good book to read?"),
    AIMessage(content="Yes! 'Atomic Habits' by James Clear is an excellent read on productivity and self-improvement."),
    HumanMessage(content="Thanks, I'll check that out."),
]

output = graph.invoke({"messages": messages})


for m in output["messages"]:
    m.pretty_print()



# the first solution to handle low message context is we have to only left the latest messages
