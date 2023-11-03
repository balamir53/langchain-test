__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import hub

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain


def run():
    dotenv.load_dotenv()
    
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo")
    
    question = "What is the meaning of life?"
    result = llm.predict(question)
    print(result)


if __name__ == "__main__":
    run()
