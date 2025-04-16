import os
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import HfApi
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# === Environment Setup ===
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv(find_dotenv())

# === Config ===
DOCUMENTS_DIR = "Data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_PAYLOAD_SIZE = 4 * 1024 * 1024  # 4 MB limit
NAMESPACE = "medibot"

# === API Keys ===
hf_token = os.getenv("HF_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# === Authentication ===
api = HfApi()
user_info = api.whoami(token=hf_token)
print("✅ Logged into Hugging Face as:", user_info.get("name", user_info))

# === Pinecone Setup ===
pc = Pinecone(api_key=pinecone_api_key)

if pinecone_index_name not in pc.list_indexes().names():
    print(f"[!] Index '{pinecone_index_name}' not found. Creating...")
    pc.create_index(
        name=pinecone_index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

pinecone_index = pc.Index(pinecone_index_name)

# === Load PDFs ===
def load_all_pdfs(directory):
    all_docs = []
    pdf_files = Path(directory).glob("*.pdf")
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            print(f"[✔] Loaded with PyPDF: {pdf_path.name}")
        except Exception:
            try:
                loader = PyMuPDFLoader(str(pdf_path))
                docs = loader.load()
                print(f"[✔] Loaded with PyMuPDF: {pdf_path.name}")
            except Exception as e:
                print(f"[✘] Failed to load {pdf_path.name}", e)
                continue
        all_docs.extend(docs)
    return all_docs

documents = load_all_pdfs(DOCUMENTS_DIR)
print("📄 Total documents loaded:", len(documents))

# === Chunking ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = text_splitter.split_documents(documents)
print("✂️ Total chunks created:", len(chunks))

# === Embedding Model ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Upload to Pinecone with Payload Limit Handling ===
def batched_upload(chunks, batch_token_limit=MAX_PAYLOAD_SIZE):
    ids = []
    embeddings = []
    metadatas = []
    texts = []
    batch_size = 0
    current_batch = []

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="🔁 Uploading"):
        metadata = chunk.metadata
        text = chunk.page_content
        id_str = f"doc-{i}"

        vector = embedding_model.embed_query(text)
        text_size = len(text.encode("utf-8"))

        item = {
            "id": id_str,
            "values": vector,
            "metadata": {"text": text, **metadata}
        }

        item_size = len(json.dumps(item).encode("utf-8"))

        if batch_size + item_size > batch_token_limit:
            pinecone_index.upsert(vectors=current_batch, namespace=NAMESPACE)
            current_batch = []
            batch_size = 0

        current_batch.append(item)
        batch_size += item_size

    # Final batch
    if current_batch:
        pinecone_index.upsert(vectors=current_batch, namespace=NAMESPACE)

batched_upload(chunks)

print(f"[✅] Upload to Pinecone complete: {pinecone_index_name} | Namespace: {NAMESPACE}")






# Step 1: Load raw PDF(s)
#import os
#from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
"""
#DATA_PATH = "Data/"

#def load_pdf_files(data):
#    documents = []
#    for filename in os.listdir(data):
#        if filename.endswith(".pdf"):
#            file_path = os.path.join(data, filename)
#            try:
                # Try loading with PyPDFLoader
#                loader = PyPDFLoader(file_path)
#                docs = loader.load()
#                print(f"[✔] Loaded with PyPDF: {filename}")
#            except Exception as e:
#                print(f"[!] PyPDF failed on {filename}: {e}")
#               try:
#                    # Fallback to PyMuPDFLoader
#                    loader = PyMuPDFLoader(file_path)
#                    docs = loader.load()
#                    print(f"[✔] Loaded with PyMuPDF: {filename}")
#                except Exception as e2:
                    print(f"[✘] Failed to load {filename} with both loaders: {e2}")
                    docs = []
            documents.extend(docs)
    return documents

documents = load_pdf_files(data=DATA_PATH)
print("Total pages loaded from all PDFs:", len(documents))


 #Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Length of text chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
Dbase_FAISS_PATH = "vectorstore/DB_faiss"
db = FAISS.from_documents(text_chunks,embedding_model)
db.save_local(Dbase_FAISS_PATH)
"""