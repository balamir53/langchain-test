import os
from dotenv import load_dotenv, find_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory

load_dotenv(find_dotenv())

llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"])
llm.temperature = 0.1

# prompts = ["The opposite of hot is", "the opposite of cold is"]
# llm_results = llm._generate(prompts)

# print(llm_results.generations[0][0].text)
# print(llm_results.generations[1][0].text)


# urls = [
#     "https://www.linkedin.com/pulse/transformers-without-pain-ibrahim-sobh-phd/",
# ]

# loader = [UnstructuredURLLoader(urls=urls)]
# index = VectorstoreIndexCreator(
#     embedding=GooglePalmEmbeddings(),
#     text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# ).from_loaders(loader)

# memory = ConversationSummaryMemory(
#         llm=llm, memory_key="chat_history", return_messages=True
#     )

# chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=index.vectorstore.as_retriever(),
#     input_key="question",
#     memory=memory
# )

# answer = chain.run('What is machine translation?')
# print(answer)


class PdfRead:
    """
    This class initializes pdfloaders, textsplitters, vectorstore, retriver,
    memory and langchain chain.
    """

    def __init__(self, folder_path) -> None:
        self.pdf_folder_path = folder_path
        self.pdf_loaders = PyPDFLoader(self.pdf_folder_path)

        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.splits = self.text_splitter.split_documents(self.pdf_loaders.load())
        self.vectorstore = Chroma.from_documents(
            documents=self.splits, embedding=GooglePalmEmbeddings()
        )
        self.retriever = self.vectorstore.as_retriever()

        self.memory = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history", return_messages=True
        )
        self.pdf_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            input_key="question",
            memory=self.memory,
        )

    def history(self, question):
        """
        returns result of llm query with question taken from user.
        """
        return self.pdf_chain.run(question)

    # pdf_answer = pdf_chain.run("What are GANs?")
    # print(answer)
