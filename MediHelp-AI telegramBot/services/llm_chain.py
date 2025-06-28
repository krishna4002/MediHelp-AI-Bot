import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Environment Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Use OpenRouter key

# Embeddings & Vector Store
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    namespace="medibot",
    pinecone_api_key=PINECONE_API_KEY
)

# Custom Prompt
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
Provide a detailed and comprehensive response, explaining concepts thoroughly.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Only provide information from the given context.

Context: {context}
Question: {question}

Start the answer directly and ensure it is well-explained and sufficiently detailed.
"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# LLM via OpenRouter
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",  # Use OpenRouter-style model name
    openai_api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.5,
    max_tokens=1024,
)

# Conversational Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 1}),
    return_source_documents=True,
    combine_docs_chain_kwargs={'prompt': prompt}
)

def get_qa_chain():
    return qa_chain
