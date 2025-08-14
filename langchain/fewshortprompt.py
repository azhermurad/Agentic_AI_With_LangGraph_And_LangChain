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
    prefix=  "You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run. Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only! Here is the schema information{schema} Below are a number of examples of questions and their corresponding Cypher queries.",
    suffix="Questions: {question}\n Cypher query:",
    input_variables=["question"],
)

print(
    prompt.invoke({"question": "Who was the father of Mary Ball Washington?","schema":"addsadasdsa"}).to_string()
)
