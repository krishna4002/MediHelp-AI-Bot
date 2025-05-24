import os
import warnings
from dotenv import load_dotenv
import tensorflow as tf

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

# --- Env Setup ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()

# --- Pinecone Credentials ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = "medibot"

# --- OpenRouter API Key ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Ensure loss function compatibility (if needed) ---
loss_function = tf.compat.v1.losses.sparse_softmax_cross_entropy
print("Loss function initialized correctly:", loss_function)

# --- Load LLM from OpenRouter ---
def load_llm_openrouter():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        model="mistralai/mistral-7b-instruct",
        temperature=0.5,
        max_tokens=512
    )

# --- Custom Prompt ---
def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
Provide a detailed and comprehensive response, explaining concepts thoroughly.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Only provide information from the given context.

Context: {context}
Question: {question}

Start the answer directly and ensure it is well-explained and sufficiently detailed.
"""

# --- Load Pinecone VectorStore ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    namespace=PINECONE_NAMESPACE,
    pinecone_api_key=PINECONE_API_KEY,
)

# --- Setup Retrieval Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm_openrouter(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
)

# --- Run query ---
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("RESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:")
for i, doc in enumerate(response["source_documents"], start=1):
    print(f"--- Document {i} ---")
    print(doc.page_content)
