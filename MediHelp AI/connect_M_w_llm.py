import os
import warnings
import tensorflow as tf
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

# --- Env Setup ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()

# --- Hugging Face & Pinecone Credentials ---
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = "medibot"  # set your namespace

# --- Ensure loss function compatibility (if needed) ---
loss_function = tf.compat.v1.losses.sparse_softmax_cross_entropy
print("Loss function initialized correctly:", loss_function)

# --- Load LLM from Hugging Face ---
def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
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
    llm=load_llm(HUGGINGFACE_REPO_ID),
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
