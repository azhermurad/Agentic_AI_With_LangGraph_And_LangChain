import os
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
import streamlit as st
# loading env varables

from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM Album LIMIT 2;"))

# 
mysql_host = 3306
mysql_pwd = "password"
mysql_user = "root"
mysql_db = "bookstore"

db = SQLDatabase.from_uri(f"mysql+pymysql://{mysql_user}:{mysql_pwd}@localhost:{mysql_host}/{mysql_db}")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")

# tooklit 

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(llm,toolkit=toolkit, agent_type='tool-calling', verbose=True)

print(agent_executor.invoke("how many books are there in book table?"))