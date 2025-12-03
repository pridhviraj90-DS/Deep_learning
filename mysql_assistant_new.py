import getpass
import urllib.parse
import pymysql
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ------------------------------------------------------
# 1. GET MYSQL CONNECTION
# ------------------------------------------------------
def get_mysql_connection_params():
    print("⚙️ Enter MySQL connection details:")
    host = input("Host (e.g., 127.0.0.1): ")
    user = input("Username: ")
    password = getpass.getpass("Password: ")
    database = input("Database name: ")
    return host, user, password, database


host, user, password, database = get_mysql_connection_params()


# ------------------------------------------------------
# 2. CONNECT WITH PYMysql
# ------------------------------------------------------
try:
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        cursorclass=pymysql.cursors.DictCursor
    )
    cursor = conn.cursor()
    print(f"✅ Connected successfully to MySQL database '{database}'!")
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    exit(1)


# ------------------------------------------------------
# 3. INITIALIZE LLM
# ------------------------------------------------------
llm = OllamaLLM(model="llama3")
parser = StrOutputParser()
print("🤖 LLM loaded OK")


# ------------------------------------------------------
# 4. SQL PROMPT
# ------------------------------------------------------
sql_prompt = ChatPromptTemplate.from_template(
    """
You are an expert MySQL assistant.

User request:
{question}

Rules:
- ALWAYS respond ONLY with a valid MySQL SELECT statement.
- NO backticks.
- NO comments.
- NO explanation.
- ONLY return SQL.
- Use ONLY existing tables from MySQL.
- If user asks for anything other than SELECT, return:
SELECT 'Invalid — only SELECT queries supported' AS message;
"""
)


# ------------------------------------------------------
# 5. ASK DATABASE FUNCTION
# ------------------------------------------------------
def ask_database(question):

    sql_query = (sql_prompt | llm | parser).invoke({"question": question}).strip()

    print("\n🧠 Generated SQL:")
    print(sql_query)

    if not sql_query.lower().startswith("select"):
        print("\n❌ BLOCKED — The generated query is not a SELECT")
        return

    print("\n🚀 Executing query...")

    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()

        if not rows:
            print("\n⚠️ No data returned")
        else:
            print("\n📊 Query Result:")
            for row in rows:
                print(row)

    except Exception as e:
        print(f"\n❌ SQL Execution Error: {e}")
        return


# ------------------------------------------------------
# 6. SHOW TABLES BEFORE START
# ------------------------------------------------------
print("\n📂 Tables in database:")
cursor.execute(f"SHOW TABLES;")
tables = cursor.fetchall()
for t in tables:
    print(" -", list(t.values())[0])


# ------------------------------------------------------
# 7. INTERACTIVE LOOP
# ------------------------------------------------------
print("\nAsk a question (type 'exit' to quit):")

while True:
    question = input("\nYour question: ")
    if question.lower() == "exit":
        print("👋 Goodbye!")
        break
    ask_database(question)
