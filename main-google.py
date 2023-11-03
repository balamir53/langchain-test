__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.chat_models import ChatVertexAI

import pprint
import google.generativeai as palm
from load_creds import load_creds


def run():
    dotenv.load_dotenv()

    # creds = load_creds()

    # load a text from web
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

    # split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(loader.load())

    # Embed and store splits
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo")
    llm = ChatVertexAI()

    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    question = "What is Task Decomposition?"
    result = qa(question)
    print(result["answer"])


if __name__ == "__main__":
    run()
