from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import START, END, StateGraph


# this is the custom reducer to make the state update 

def CustomReduicer(previous_state, updated_state):
    print("aaa",previous_state, updated_state)
    return [1000]
    
    
class State(TypedDict):
    books: Annotated[list[int], add]
    count: Annotated[list[int], CustomReduicer]

    #

def node_1(state: State):
    print(state)
    # this reducer can update the state of the graph
    return {"books": [state["books"][-1] + 1],"count":[44]}


def node_2(state: State):
    # this reducer can update the state of the graph
    print(state)
    
    return {"books": [state["books"][-1] + 2]}


def node_3(state: State):
    print(state)
    # this reducer can update the state of the graph
    return {"books": [state["books"][-1]+4]}


builder = StateGraph(State)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# logic


# now we have to

#

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2",END)
builder.add_edge("node_3",END)


# now we have to compile this function to get the response from the nodes
graph = builder.compile()


from langgraph.errors import InvalidUpdateError

try:
    # we have to handle the error of the agent
    result = graph.invoke({"books": [1],"count":[2]})
    print(result, "this is the output from the agent ")
except InvalidUpdateError as e:
    # we have to output the error here
    print(e)
