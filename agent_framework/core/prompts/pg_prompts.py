from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent_framework.core.states.pg_states import PostgresDatabaseState

pg_table_information_extractor = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an SQL expert specializing in analyzing {dialect} schema structures, "
            "helping users understand what information this table contains."
            "consider foreign table information if this table contains foreign keys.."
            "{format_instructions}",
        ),
        (
            "human",
            """
            Given an input question, 
            first create a syntactically correct {dialect} query to run, 
            then look at the results of the query and return the answer. 

            Only use the following tables: 
            main table: {table} with columns: {columns}, 

            primary key information: {primary_key}, 
            foreign key information: {relationship_desc}, 

            related tables: 
            {related_tables_desc}

            Question: {question}.
            """,
        ),
    ]
)
