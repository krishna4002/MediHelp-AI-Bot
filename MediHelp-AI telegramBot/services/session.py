import os
import json

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

def get_session_path(user_id):
    return os.path.join(SESSIONS_DIR, f"{user_id}.json")

def load_session(user_id):
    path = get_session_path(user_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_session(user_id, messages):
    path = get_session_path(user_id)
    with open(path, "w") as f:
        json.dump(messages, f, indent=4)

def export_session_txt(user_id):
    messages = load_session(user_id)
    lines = []
    for m in messages:
        lines.append(f"{m['role'].capitalize()}:\n{m['content']}\n")
    return "\n".join(lines)

def export_session_json(user_id):
    messages = load_session(user_id)
    return json.dumps(messages, indent=4)
