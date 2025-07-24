import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent,create_tool_calling_agent
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor 


# loading env varables
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")


# embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

# load document 
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()


# split document into chunks
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)


# vector store
vector = FAISS.from_documents(documents, embeddings)


# retriver
retriever = vector.as_retriever()


# tools

retriever_tool = create_retriever_tool(
    retriever, 
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
)
arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=400)
)

tools = [wikipedia_tool, arxiv_tool,retriever_tool]


# Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/react")
prompt = hub.pull("hwchase17/openai-functions-agent")



# create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)


response = agent_executor.invoke({"input": "how can langsmith help with testing?"})

print(response)
