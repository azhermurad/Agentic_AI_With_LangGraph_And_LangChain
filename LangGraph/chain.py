from pprint import pprint
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# loading env varables
from dotenv import load_dotenv

load_dotenv()


# load environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")


messages = [
    AIMessage(content="so you said that your researching ocean mammals?", name="model")
]


messages.extend([HumanMessage("yes that's right", name="lance")])
messages.extend(
    [AIMessage(content="great! what you want to learn about?", name="model")]
)
messages.extend(
    [
        HumanMessage(
            content="i wnat to learn about the best places of where mammals are live? ",
            name="lance",
        )
    ]
)


# for x in messages:
#     x.pretty_print()


# print(llm.invoke(messages))


# tool calling


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


lm_with_tools = llm.bind_tools([multiply])
result = lm_with_tools.invoke(
    [HumanMessage(content="What is 2 multiplied by 3?", name="lance")]
)
# print(result)
# print(result.tool_calls)


# in langgraph we have to make the state of the graph, node(reducers), edges,


from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState, StateGraph, START, END


# define the state
class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class PrebuildMessageState(MessagesState):
    pass


# nodes


def tool_calling_llm(state: PrebuildMessageState):
    print("message in the node", state)
    return {"messages": [lm_with_tools.invoke(state["messages"])]}


# building graph

builder = StateGraph(PrebuildMessageState)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

graph = builder.compile()


result = graph.invoke({"messages": HumanMessage(content="hello")})
print(result)

result = graph.invoke({"messages": HumanMessage(content="What is 2 multiplied by 3?")})
print(result)



# with open("diagram.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())
# print("Saved diagram.png")
