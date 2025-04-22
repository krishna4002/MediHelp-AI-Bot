# 🏥 MediHelp AI

**MediHelp AI** is an intelligent, context-aware medical chatbot powered by large language models (LLMs) and Pinecone vector search. It allows users to interact with medical data, ask complex queries, and retrieve detailed, source-backed answers.

## 🚀 Features

Let me know if you'd like it downloaded as a file or if you want your actual name included under Contributors!
- 🤖 Integration with Hugging Face LLMs (e.g., Mistral 7B).
- 📚 Contextual document retrieval via Pinecone vector store.
- 🧠 Memory-enabled chat sessions with save/load/edit/delete functionality.
- 📁 Export chat history as `.txt` files.

## 📂 Project Structure

```bash
├── MediHelp_AI.py            # Streamlit app file
├── connect_M_w_llm.py        # Command-line QA script (CLI interface)
├── .env                      # Environment variables (API keys, Pinecone setup)
├── chat_sessions/            # Folder for saved user sessions
```

## 🧪 Requirements

- Python 3.10+
- Hugging Face Transformers
- Streamlit
- LangChain
- Pinecone
- fpdf

Install dependencies with:

```bash
pip install -r requirements.txt
```

> You can generate `requirements.txt` with:
```bash
pip freeze > requirements.txt
```

## 🔧 Setup

1. Create a `.env` file and fill it with your credentials:

```env
HF_TOKEN="your_huggingface_api_token"
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_ENV="your_pinecone_environment"
PINECONE_INDEX_NAME="your_pinecone_index_name"
```

2. Ensure Pinecone is configured with a suitable index using `"sentence-transformers/all-MiniLM-L6-v2"` as the embedding model.

## 💬 Running the Chatbot

Run the Streamlit interface with:

```bash
streamlit run MediHelp_AI.py
```

Or use the terminal script (`connect_M_w_llm.py`) for CLI interaction:

```bash
python connect_M_w_llm.py
```

## ✍️ Custom Prompt

All interactions use a strict prompt template that:

- Stresses factuality.
- Encourages detailed explanations.
- Disallows hallucinations or fabrications.

## 📝 Session Management

- Sessions are stored in `chat_sessions/` as JSON files.
- Rename, delete, or export chat logs through the UI sidebar.

## 📄 Export

Chats can be exported in `.txt` format from the sidebar.

## 📌 Notes

- You can use local models or Hugging Face Inference Endpoints.
- LLM and retriever are flexible and modular—can be swapped easily.

## 👥 Contributors

- **Krishnagopal Jay**  
- **Ritam Koley**  
- **Jit Mandal**  
- **Anwesha Das** 
