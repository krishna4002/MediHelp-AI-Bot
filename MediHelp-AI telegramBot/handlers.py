import asyncio
from aiogram import Router, types
from aiogram.filters import Command
from services.llm_chain import get_qa_chain
from services.session import load_session, save_session, export_session_txt, export_session_json

router = Router()

user_histories = {}

@router.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Welcome to MediHelp AI Bot!\nAsk me any medical question.")

@router.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer("/new - Start new session\n/history - View past questions\n/edit - Edit a question\n/download txt - Download chat as TXT\n/download json - Download chat as JSON")

@router.message(Command("new"))
async def cmd_new(message: types.Message):
    user_id = str(message.from_user.id)
    user_histories[user_id] = []
    save_session(user_id, [])
    await message.answer("Started a new session.")

@router.message(Command("history"))
async def cmd_history(message: types.Message):
    user_id = str(message.from_user.id)
    session = load_session(user_id)
    if not session:
        await message.answer("No history yet.")
        return
    history = "\n".join([f"{idx+1}. {m['content']}" for idx, m in enumerate(session) if m['role'] == 'user'])
    await message.answer(f"Previous Questions:\n\n{history}")

@router.message(Command("edit"))
async def cmd_edit(message: types.Message):
    await message.answer("Editing not implemented yet. Coming soon!")

@router.message(Command("download"))
async def cmd_download(message: types.Message):
    user_id = str(message.from_user.id)
    args = message.text.split()
    if len(args) < 2:
        await message.answer("Please specify format: /download txt OR /download json")
        return
    format = args[1]
    if format == "txt":
        content = export_session_txt(user_id)
        await message.answer_document(types.BufferedInputFile(content.encode(), filename=f"session_{user_id}.txt"))
    elif format == "json":
        content = export_session_json(user_id)
        await message.answer_document(types.BufferedInputFile(content.encode(), filename=f"session_{user_id}.json"))
    else:
        await message.answer("Invalid format. Use txt or json.")

@router.message()
async def handle_query(message: types.Message):
    user_id = str(message.from_user.id)
    question = message.text
    session = load_session(user_id)

    history_pairs = []
    for i in range(0, len(session) - 1, 2):
        if session[i]["role"] == "user" and session[i+1]["role"] == "assistant":
            history_pairs.append((session[i]["content"], session[i+1]["content"]))

    chain = get_qa_chain()
    response = await asyncio.to_thread(chain, {
        "question": question,
        "chat_history": history_pairs[-5:]
    })

    answer = response["answer"]
    sources = "\n\n".join([
        f"- Source: {doc.metadata.get('source', 'Unknown')} | Title: {doc.metadata.get('title', 'N/A')}"
        for doc in response["source_documents"]
    ])
    final_answer = answer + "\n\n---\n\nSources:\n" + sources

    await message.answer(final_answer)

    # Save to session
    session.append({"role": "user", "content": question})
    session.append({"role": "assistant", "content": final_answer})
    save_session(user_id, session)
