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
from langgraph.checkpoint.memory import InMemorySaver


# loading env varables
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")


class State(MessagesState):
    summary: str


def summarize_conversation(state: State):

    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:

        # A summary already exists
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def chatbot(state: State):
    # get the summary if exit
    summary = state.get("summary", "")
    if summary:
        print("hello")
        system_message = f"summary of converstion ealier {summary}"

        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = llm.invoke(messages)
    return {"messages": response}


# conductional edge


def should_summary(state: State):
    """Return the next node to execute."""

    if len(state["messages"]) > 2:
        return "summarize_conversation"
    return END


# add node
builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("summarize_conversation", summarize_conversation)


# edges

builder.add_edge(START, "chatbot")
# builder.add_conditional_edges("chatbot",should_summary)
builder.add_conditional_edges(
    "chatbot",
    should_summary,
    {"summarize_conversation": "summarize_conversation", END: END},
)
builder.add_edge("summarize_conversation", END)


memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

with open("diagram.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("Saved diagram.png")

config = {"configurable": {"thread_id": "1"}}

input_message = HumanMessage(content="Hi, I'am azher ali")
output = graph.invoke({"messages": input_message}, config)

for m in output["messages"]:
    m.pretty_print()
    
    
input_message = HumanMessage(content="what is my name")
output = graph.invoke({"messages": input_message}, config)

for m in output["messages"]:
    m.pretty_print()



print(graph.get_state(config).values.get("summary"))


