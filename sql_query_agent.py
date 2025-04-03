import os
import re
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.chains import create_sql_query_chain

DB_HOST = "ENTER YOUR HOSTNAME"
DB_NAME = "ENTER YOUR DATABASE NAME"
DB_USER = "ENTER YOUR DB USER"
DB_PASSWORD = os.getenv("DB_PASSWORD", "YOUR_DB_PASSWORD")

endpoint = os.getenv("ENDPOINT_URL", "YOUR_AZURE_ENDPOINT")
deployment = os.getenv("DEPLOYMENT_NAME", "YOUR_AZURE_DEPLOYMENT")

def extract_sql_query(text):
    """Extract SQL query from AI-generated text."""
    pattern = r"SELECT[\s\S]*?(?=```|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group().strip() if match else None

def azure_sql_gm3(question: str):
    """Generates and executes an SQL query using Azure OpenAI."""
    db_uri = f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?driver=ODBC+Driver+18+for+SQL+Server"

    try:
        db_engine = create_engine(db_uri)
        db = SQLDatabase(db_engine, view_support=True, schema="dbo")

        llm = AzureChatOpenAI(azure_endpoint=endpoint, azure_deployment=deployment, openai_api_version="2024-08-01-preview")
        generate_query = create_sql_query_chain(llm, db)
        query = generate_query.invoke({'question': question})
        extracted_query = extract_sql_query(query)

        if extracted_query:
            execute_query = QuerySQLDatabaseTool(db=db)
            return execute_query.invoke(extracted_query)

    except SQLAlchemyError as db_err:
        print(f"❌ SQL Execution Error: {db_err}")

    return None
