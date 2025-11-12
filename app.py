import os, uuid, time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__, static_url_path='/static')
CORS(app)  # allow embedding in Qualtrics iFrame

SESSIONS = {}  # in-memory; use a DB in production (SQLite/Postgres)

def load_persona():
    with open("prompts/persona.md", "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT = load_persona()

@app.route("/health")
def health():
    return {"ok": True}

@app.route("/new_session", methods=["POST"])
def new_session():
    sid = str(uuid.uuid4())
    meta = request.json or {}
    SESSIONS[sid] = {
        "created": time.time(),
        "meta": meta,           # e.g., learner_id, case_id, OSCE station #
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi, I’m Brenda Mae. How can I help today?"}
        ],
        "complete": False
    }
    return jsonify({"session_id": sid, "greeting": SESSIONS[sid]["messages"][-1]["content"]})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    sid = data.get("session_id")
    user_msg = data.get("message","").strip()

    if sid not in SESSIONS:
        return jsonify({"error":"invalid session"}), 400
    if SESSIONS[sid]["complete"]:
        return jsonify({"reply":"This session is complete."})

    # Record user turn
    SESSIONS[sid]["messages"].append({"role":"user", "content": user_msg})

    # Call the model
    completion = client.chat.completions.create(
        model="gpt-5.1-mini",  # small, fast; swap to your preferred model
        temperature=0.6,
        messages=SESSIONS[sid]["messages"][:20]  # keep context bounded
    )
    reply = completion.choices[0].message.content

    # Check completion trigger
    if "SESSION_COMPLETE" in reply.upper():
        SESSIONS[sid]["complete"] = True
        reply += "\n\nThank you. Please let your examiner know you’re done."

    SESSIONS[sid]["messages"].append({"role":"assistant","content": reply})

    return jsonify({"reply": reply})

@app.route("/transcript/<sid>", methods=["GET"])
def transcript(sid):
    if sid not in SESSIONS: return jsonify({"error":"invalid session"}), 400
    # Redact system prompt for assessors; include user/assistant turns
    turns = [m for m in SESSIONS[sid]["messages"] if m["role"] != "system"]
    return jsonify({"meta": SESSIONS[sid]["meta"], "turns": turns, "complete": SESSIONS[sid]["complete"]})

# Serve the minimal chat UI (for quick testing and embedding)
@app.route("/")
def index():
    return send_from_directory("static", "chat.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
