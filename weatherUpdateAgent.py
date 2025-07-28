import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import streamlit as st


load_dotenv()
# load environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")


def get_weather(city:str)-> str:
    """Get the weather for a city."""
    return f"it's always sunny in {city}!"



agent = create_react_agent(
    model = llm,
    tools= [get_weather],
    prompt="You are a helpful assistant"
)


# run the agent
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
# )

# print(response)

# Configure an LLM¶
# To configure an LLM with specific parameters, such as temperature, use init_chat_model:
from langchain.chat_models import init_chat_model

model  = init_chat_model("groq:llama-3.1-8b-instant", temperature=0.5)


agent = create_react_agent(
    model = model,
    tools= [get_weather],
    # A static prompt that never changes
    prompt="Never answer questions about the weather."
)


res = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(res)

# 4. Add a memory 


from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver



checkpoint = InMemorySaver()

agent = create_react_agent(
    model = model,
    tools= [get_weather],
    checkpointer=checkpoint
)

config = {"configurable": {"thread_id": "1"}}

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config=config,


)

ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config
)




st.write(sf_response)





