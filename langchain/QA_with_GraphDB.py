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
  

from langchain_neo4j import Neo4jGraph

# Import movie information
movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
MERGE (m:Movie {id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""
# setup the connection  with  neo4j database
graph = Neo4jGraph(enhanced_schema=True,url= os.getenv("NEO4J_URI"))
# graph.query(movies_query)

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

example_prompt = PromptTemplate.from_template(
    "Question: {question}\nCypher: {query}"
)


examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {{title: 'Casino'}})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {{title: 'Schindler''s List'}})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": (
            "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), "
            "(a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) "
            "WHERE g1.name = 'Comedy' AND g2.name = 'Action' "
            "RETURN DISTINCT a.name"
        ),
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": (
            "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) "
            "WHERE a.name STARTS WITH 'John' "
            "WITH d, COUNT(DISTINCT a) AS JohnsCount "
            "WHERE JohnsCount >= 3 "
            "RETURN d.name"
        ),
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": (
            "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) "
            "RETURN a.name, COUNT(m) AS movieCount "
            "ORDER BY movieCount DESC LIMIT 1"
        ),
    },
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=  "You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run. Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only! Here is the schema information {schema} Below are a number of examples of questions and their corresponding Cypher queries.",
    suffix="Questions: {question}\n Cypher query:",
    input_variables=["question"],
)


from langchain_neo4j import GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(
    graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True,
    cypher_prompt=prompt
)

question2 = "Find the actor with the highest number of movies in the database."
print(chain.invoke({"query":question2}))


