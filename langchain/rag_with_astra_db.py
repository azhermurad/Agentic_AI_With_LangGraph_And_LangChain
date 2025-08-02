import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore

# loading env varables
from dotenv import load_dotenv

load_dotenv()


# load environment variables
os.environ["ASTRA_DB_API_ENDPOINT"] = os.getenv("ASTRA_DB_API_ENDPOINT")
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


vector_store_explicit_embeddings = AstraDBVectorStore(
    collection_name="astra_vector_langchain",
    embedding=embeddings,
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    autodetect_collection=True,
    content_field="my_document_text_field"
)

vector_store = vector_store_explicit_embeddings
from langchain_core.documents import Document

documents_to_insert = [
    Document(
        page_content="ZYX, just another tool in the world, is actually my agent-based superhero",
        metadata={"source": "tweet"},
        id="entry_00",
    ),
    Document(
        page_content="I had chocolate chip pancakes and scrambled eggs "
        "for breakfast this morning.",
        metadata={"source": "tweet"},
        id="entry_01",
    
    ),
    Document(
        page_content="The weather forecast for tomorrow is cloudy and "
        "overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
        id="entry_02",
    ),
    Document(
        page_content="Building an exciting new project with LangChain "
        "- come check it out!",
        metadata={"source": "tweet"},
        id="entry_03",
    ),
    Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
        id="entry_04",
    ),
    Document(
        page_content="Thanks to her sophisticated language skills, the agent "
        "managed to extract strategic information all right.",
        metadata={"source": "tweet"},
        id="entry_05",
    ),
    Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
        id="entry_06",
    ),
    Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
        id="entry_07",
    ),
    Document(
        page_content="LangGraph is the best framework for building stateful, "
        "agentic applications!",
        metadata={"source": "tweet"},
        id="entry_08",
    ),
    Document(
        page_content="The stock market is down 500 points today due to "
        "fears of a recession.",
        metadata={"source": "news"},
        id="entry_09",
    ),
    Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
        id="entry_10",
    ),
]


# split document into chunks
# documents = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200
# ).split_documents(docs)
# vector_store.add_documents(documents=documents_to_insert)

retriever = vector_store.as_retriever()

print(retriever.invoke("what is langchain?"))




