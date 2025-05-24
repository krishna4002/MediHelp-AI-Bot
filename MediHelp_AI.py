import os
import re
import json
import streamlit as st
from fpdf import FPDF
from datetime import datetime

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

#DB_FAISS_PATH = "vectorstore/DB_faiss"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_NAMESPACE = "medibot"
CHAT_SESSIONS_PATH = "chat_sessions"
os.makedirs(CHAT_SESSIONS_PATH, exist_ok=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_model,
        namespace=PINECONE_NAMESPACE,
        pinecone_api_key=PINECONE_API_KEY
    )
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm_openrouter():
    return ChatOpenAI(
        temperature=0.5,
        model="mistralai/mistral-7b-instruct",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY")
    )

def format_numbered_list(text):
    return re.sub(r"(\d+\.)", r"\n\1", text)

def save_session(name, messages):
    with open(f"{CHAT_SESSIONS_PATH}/{name}.json", "w") as f:
        json.dump(messages, f)

def load_session(name):
    try:
        with open(f"{CHAT_SESSIONS_PATH}/{name}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def delete_session(name):
    path = f"{CHAT_SESSIONS_PATH}/{name}.json"
    if os.path.exists(path):
        os.remove(path)

def rename_session(old_name, new_name):
    os.rename(f"{CHAT_SESSIONS_PATH}/{old_name}.json", f"{CHAT_SESSIONS_PATH}/{new_name}.json")

def get_all_sessions():
    return [f.replace(".json", "") for f in os.listdir(CHAT_SESSIONS_PATH) if f.endswith(".json")]

def generate_txt(messages):
    return "\n\n".join(f"{msg['role'].capitalize()}:\n{msg['content']}" for msg in messages)



def main():
    st.set_page_config(page_title="Chatbot", layout="wide")
    #st.title("Welcome to MediHelp AI - Your Smart Medical Assistant!")
    st.markdown("## Welcome to MediHelp AI - Your Smart Medical Assistant!")

    # Sidebar - Session Handling
    with st.sidebar:
        st.subheader("Chat Sessions")
        sessions = get_all_sessions()

        if "current_session" not in st.session_state:
            st.session_state.current_session = sessions[0] if sessions else "Session 1"
            st.session_state.messages = load_session(st.session_state.current_session)

        if st.button("➕ New Chat"):
            new_name = f"Session {len(sessions) + 1}"
            st.session_state.current_session = new_name
            st.session_state.messages = []
            save_session(new_name, [])

        selected = st.selectbox("Select Session", sessions, index=sessions.index(st.session_state.current_session) if st.session_state.current_session in sessions else 0)
        if selected != st.session_state.current_session:
            st.session_state.current_session = selected
            st.session_state.messages = load_session(selected)

        new_name = st.text_input("Rename session", placeholder="Enter new session name")
        if st.button("📝 Rename") and new_name and new_name != st.session_state.current_session:
            rename_session(st.session_state.current_session, new_name)
            st.session_state.current_session = new_name
            st.session_state.messages = load_session(new_name)

        if st.button("🗑 Delete"):
            delete_session(st.session_state.current_session)
            sessions = get_all_sessions()
            st.session_state.current_session = sessions[0] if sessions else "Session 1"
            st.session_state.messages = load_session(st.session_state.current_session)

        st.markdown("---")
        st.subheader("Export Chat")
        export_format = st.selectbox("Format", ["TXT"])
        if st.button("⬇ Download"):
            if export_format == "TXT":
                content = generate_txt(st.session_state.messages)
                st.download_button("Download TXT", content, file_name=f"{st.session_state.current_session}.txt")


    # Display chat messages and handle edit
    messages_to_remove = []
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                if f"edit_mode_{idx}" not in st.session_state:
                    st.session_state[f"edit_mode_{idx}"] = False

                if st.session_state[f"edit_mode_{idx}"]:
                    new_text = st.text_area("Edit your prompt", value=message["content"], key=f"input_{idx}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save & Regenerate", key=f"save_{idx}"):
                            st.session_state.messages[idx]["content"] = new_text
                            st.session_state[f"edit_mode_{idx}"] = False
                            if idx + 1 < len(st.session_state.messages) and st.session_state.messages[idx + 1]["role"] == "assistant":
                                del st.session_state.messages[idx + 1]  # Remove old assistant response
                            st.session_state.regenerate_index = idx
                            save_session(st.session_state.current_session, st.session_state.messages)
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_{idx}"):
                            st.session_state[f"edit_mode_{idx}"] = False
                            st.rerun()
                else:
                    st.markdown(message["content"])
                    if st.button("✏ Edit", key=f"edit_{idx}"):
                        st.session_state[f"edit_mode_{idx}"] = True
                        st.rerun()
            else:
                st.markdown(message["content"])

    # Handle edited prompt regeneration
    if "regenerate_index" in st.session_state:
        edited_prompt = st.session_state.messages[st.session_state.regenerate_index]["content"]
        st.chat_message("user").markdown(edited_prompt)

        custom_prompt_template = """
        Use the pieces of information provided in the context to answer the user's question.
        Provide a detailed and comprehensive response, explaining concepts thoroughly.
        If you don't know the answer, just say that you don't know. Don't make up an answer.
        Only provide information from the given context.

        Context: {context}
        Question: {question}

        Start the answer directly and ensure it is well-explained and sufficiently detailed.
        """

        try:
            vectorstore = get_vectorstore()
            qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=load_llm_openrouter(),
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 1}),
                    return_source_documents=True,
                    combine_docs_chain_kwargs={
                        'prompt': set_custom_prompt(custom_prompt_template),
                        'document_variable_name': 'context'
                    }
            )
            

            # History building
            history_pairs = [
                (m["content"], st.session_state.messages[i + 1]["content"])
                for i, m in enumerate(st.session_state.messages[:-1])
                if m["role"] == "user" and st.session_state.messages[i + 1]["role"] == "assistant"
            ]
            response = qa_chain({
                "question": edited_prompt,
                "chat_history": history_pairs[-5:]
            })

            result = format_numbered_list(response["answer"])
            sources = "\n\n".join([
                f"- Source: {doc.metadata.get('source', 'Unknown')} | Title: {doc.metadata.get('title', 'N/A')} \n*Page Content*: {doc.page_content}"
                for doc in response["source_documents"]
            ])

            final_output = result + "\n\n---\n\n*Source Docs:*\n\n" + sources
            st.chat_message("assistant").markdown(final_output)
            st.session_state.messages.insert(st.session_state.regenerate_index + 1, {"role": "assistant", "content": final_output})
            st.session_state[f"edit_mode_{st.session_state.regenerate_index}"] = False
            del st.session_state["regenerate_index"]
            save_session(st.session_state.current_session, st.session_state.messages)
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            del st.session_state["regenerate_index"]

    # Handle prompt input (store but delay handling until next render cycle)
    if "regenerate_index" not in st.session_state:
        prompt = st.chat_input("Ask about any Medical queries...")
        if prompt:
            st.session_state.pending_prompt = prompt
            st.rerun()

    # Handle a pending prompt (safe from skipping logic)
    if "pending_prompt" in st.session_state:
        last_prompt = st.session_state.pending_prompt

        with st.chat_message("user"):
            st.markdown(last_prompt)

        st.session_state.messages.append({"role": "user", "content": last_prompt})
        save_session(st.session_state.current_session, st.session_state.messages)

        with st.spinner("Generating response..."):
            try:
                vectorstore = get_vectorstore()
                custom_prompt_template = """
                Use the pieces of information provided in the context to answer the user's question.
                Provide a detailed and comprehensive response, explaining concepts thoroughly.
                If you don't know the answer, just say that you don't know. Don't make up an answer.
                Only provide information from the given context.

                Context: {context}
                Question: {question}

                Start the answer directly and ensure it is well-explained and sufficiently detailed.
                """

                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=load_llm_openrouter(),
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 1}),
                    return_source_documents=True,
                    combine_docs_chain_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
                )

                history_pairs = []
                for i in range(0, len(st.session_state.messages) - 1, 2):
                    if st.session_state.messages[i]["role"] == "user" and st.session_state.messages[i + 1]["role"] == "assistant":
                        history_pairs.append((st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"]))

                response = qa_chain({
                    "question": last_prompt,
                    "chat_history": history_pairs[-5:]
                })

                result = format_numbered_list(response["answer"])
                source_documents = response["source_documents"]
                formatted_sources = "\n\n".join([
                    f"- Source: {doc.metadata.get('source', 'Unknown')} | Title: {doc.metadata.get('title', 'N/A')} \n*Page Content*: {doc.page_content}"
                    for doc in source_documents
                ])
                result_to_show = result + "\n\n---\n\n*Source Docs:*\n\n" + formatted_sources

                st.chat_message("assistant").markdown(result_to_show)
                st.session_state.messages.append({"role": "assistant", "content": result_to_show})
                save_session(st.session_state.current_session, st.session_state.messages)

            except Exception as e:
                st.error(f"Error: {str(e)}")

            # Clear pending prompt and rerun
            del st.session_state.pending_prompt
            st.rerun()

if __name__ == "__main__":
    main()

