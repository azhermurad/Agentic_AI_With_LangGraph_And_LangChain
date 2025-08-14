import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# loading env varables
from dotenv import load_dotenv

load_dotenv()


# load environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer 
            the question. If you don't know the answer, say that you 
            don't know.
            
            {context}
            
            Question: {input}

            Helpful Answer:
            """,
        ),
    ]
)


# load document
loader = WebBaseLoader("https://github.com/azhermurad")
docs = loader.load()


documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)



print(documents)
# vector_store.add_documents(documents=documents_to_insert)



# db = Neo4jVector.from_documents(
#     documents,embeddings
# )

index_name = "vector"  # default index name

store = Neo4jVector.from_existing_index(
    embeddings,
    index_name=index_name,
)


retriever = store.as_retriever()


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print(rag_chain.invoke({"input": "who is azher ali?"})["answer"])



