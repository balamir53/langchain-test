'''
langchain db integration to keep memory on a local postgres DB.
'''
from langchain.memory import PostgresChatMessageHistory
from palm_langchain import PdfRead

history = PostgresChatMessageHistory(
    connection_string="postgresql://luchy:luchy@localhost:5432/postgres", session_id="foo")

FILE_PATH = "/home/beast/langchain-test/1.pdf"
pdfreader = PdfRead(FILE_PATH)

while True:
    question = str(input())
    answer = pdfreader.history(question)
    history.add_user_message(question)
    history.add_ai_message(answer)
