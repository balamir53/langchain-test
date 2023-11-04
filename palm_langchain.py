import os
from dotenv import load_dotenv, find_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


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


pdf_folder_path = "/home/beast/langchain-test/1.pdf"
pdf_loaders = PyPDFLoader(pdf_folder_path)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(pdf_loaders.load())
vectorstore = Chroma.from_documents(documents=splits, embedding=GooglePalmEmbeddings())
retriever = vectorstore.as_retriever()

pdf_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, input_key="question"
)

pdf_answer = pdf_chain.run("What are GANs?")
print(pdf_answer)
