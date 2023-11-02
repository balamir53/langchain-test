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
    # load a text from web
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    # a = loader.load()
    # split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(loader.load())

    # Embed and store splits
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Prompt
    # https://smith.langchain.com/hub/rlm/rag-prompt
    # rag_prompt = hub.pull("rlm/rag-prompt")

    # LLM
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo")
    # llm = AzureOpenAI(
    # model_name="gpt-35-turbo",
    # model_name="gpt-35-turbo",
    # )

    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    # RAG chain
    # rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm

    # rag_chain.invoke("What is Task Decomposition?")

    question = "What is Task Decomposition?"
    result = qa(question)
    print(result["answer"])


if __name__ == "__main__":
    run()
