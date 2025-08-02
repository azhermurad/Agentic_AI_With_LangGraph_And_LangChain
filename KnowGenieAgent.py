import streamlit as st
import os
from langchain.agents import AgentExecutor, create_react_agent,create_tool_calling_agent
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain import hub
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import hashlib


load_dotenv()


st.title("Multi-Source AI Knowledge Agent -- Search Wiki, Arxiv & Custom PDF Data")
st.set_page_config(layout="wide")
"""This intelligent AI agent empowers users to explore information from multiple trusted sources — including Wikipedia, Arxiv, and personalized PDF documents — all in one seamless chat interface. By combining real-time search with Retrieval-Augmented Generation (RAG), it delivers accurate, context-aware answers tailored to your needs.

Ideal for researchers, students, and professionals seeking instant insights from scholarly articles, encyclopedic knowledge, and custom data — without switching tabs.

"""

# page with


# api key
with st.sidebar:
    st.title("Settings")
    os.environ["GROQ_API_KEY"] = st.text_input(
        "API KEY OF GROQ", type="password", value=os.getenv("GROQ_API_KEY")
    )
    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    def get_file_hash(file_bytes):
        return hashlib.md5(file_bytes).hexdigest()

    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_hash = get_file_hash(file_bytes)
        if st.session_state.get("last_file_hash") != file_hash:
            print("this function only called when we change the file")
            st.session_state["last_file_hash"] = file_hash
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                uploaded_file.seek(0)
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load PDF with LangChain
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

            # load document
            loader = WebBaseLoader("https://github.com/azhermurad")
            docs = loader.load()

         

            # vector store
            vector = FAISS.from_documents(docs, embeddings)

            # retriver
            retriever = vector.as_retriever()

            st.session_state["retriever"] = retriever

            retriever_tool = create_retriever_tool(
                st.session_state.retriever,
                "personal_infomation_search",
                "Search for information about azher ali. For any questions about azher ali, you must use this tool!",
            )
        else:
            print("same file")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I'am a Agent JamXT which have power of search wikipiedia, arxvi, and your local documents, How can I help you?",
        }
    ]


# message container
msg_container = st.container()

for msg in st.session_state.messages:
    with msg_container.chat_message(msg["role"]):
        st.write(msg["content"])


if bool(os.environ["GROQ_API_KEY"]):
    if input := st.chat_input(
        "Say something!", disabled=not bool(os.environ["GROQ_API_KEY"])
    ):
        # llm model
        llm = ChatGroq(model="llama-3.1-8b-instant")

        # tools
        wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
        )
        arxiv_tool = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=400)
        )

        tools = (
            [wikipedia_tool, arxiv_tool, retriever_tool]
            if st.session_state.get("retriever")
            else [wikipedia_tool, arxiv_tool]
        )


        prompt = hub.pull("hwchase17/openai-functions-agent")
        # tool calling prompt
        
        
        
        
        # create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )

        st.session_state.messages.append({"role": "user", "content": input})
        with msg_container.chat_message(
            "user",
            avatar="https://media.licdn.com/dms/image/v2/D4D03AQELrOU4Hf6Jeg/profile-displayphoto-scale_100_100/B4DZf.QMXBHkAg-/0/1752317351587?e=1756339200&v=beta&t=aVlXfjLRnFJU2pkBU_g_nlPYTpyfQdPEtq4tX3glA_c",
        ):
            st.write(input)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.invoke(
                {"input": input}, {"callbacks": [st_callback]}
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": response["output"]}
            )
            st.write(response["output"])
else:
    st.warning("API key is missing!!!")
