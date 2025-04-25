
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    message = request.message

    if session_id not in sessions:
        sessions[session_id] = {
            "history": [
                {"role": "system", "content": (
                    "You are Doctor AI, a professional clinical assistant. "
                    "Ask one question at a time to collect enough information, "
                    "then give a percentage-based diagnosis in the end. "
                    "Speak like a calm, respectful doctor."
                )}
            ],
            "questions_asked": 0,
            "diagnosis_given": False
        }

    session = sessions[session_id]
    session["history"].append({"role": "user", "content": message})

    if session["questions_asked"] < 5:
        prompt = (
            "Based on everything the patient has said so far, ask the next most important follow-up question. "
            "Only ask one question. Do not summarize or diagnose yet."
        )
        session["history"].append({"role": "user", "content": prompt})
        session["questions_asked"] += 1
    else:
        prompt = (
            "Based on the entire conversation so far, provide a list of possible diagnoses with percentage likelihoods. "
            "Be clear, clinical, and professional in your wording."
        )
        session["history"].append({"role": "user", "content": prompt})
        session["diagnosis_given"] = True

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=session["history"]
    )

    reply = response["choices"][0]["message"]["content"]
    session["history"].append({"role": "assistant", "content": reply})

    return {"message": reply}
