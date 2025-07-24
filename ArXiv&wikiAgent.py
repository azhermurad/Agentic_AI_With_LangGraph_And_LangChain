import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper




# loading env varables
from dotenv import load_dotenv
load_dotenv()

# load environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")


wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=400))
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=400))

tools = [wikipedia_tool, arxiv_tool]
from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)


from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

agent_executor.invoke({"input": "what is the full form of NATO"})
