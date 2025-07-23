import os
from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun

# https://python.langchain.com/docs/how_to/agent_executor/ 

# loading env varables
from dotenv import load_dotenv
load_dotenv()


# load environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")

# # embedding model
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2", show_progress=True
# )


from langchain_core.tools import tool



search = DuckDuckGoSearchRun()
tools = [search]

# model_with_tools = llm.bind_tools(tools)

# response = model_with_tools.invoke("hi!")

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

# response = model_with_tools.invoke("What is the weather in New York?")


# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

# create the agent

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
# hwchase17/react

# prompt from langchain hub
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm,tools,prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True,)

response = agent_executor.invoke({"input": "CAPITAL OF PAKISTAN"})

print(response)

